import sys
import numpy as np
import datetime
import json
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
import warnings

# ===================================================================
# SUPRESSÃO DE AVISOS E BLOQUEIO DE JANELAS EXTERNAS DO MATPLOTLIB
# ===================================================================
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.show = lambda *args, **kwargs: None

# ===================================================================
# IMPORTS
# ===================================================================
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                                 QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                                 QLineEdit, QComboBox, QCheckBox, QPushButton, 
                                 QTableWidget, QTableWidgetItem, QHeaderView, 
                                 QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox,
                                 QFileDialog, QFrame)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    import matplotlib
    matplotlib.use('Qt5Agg') 
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

try:
    import nidaqmx
    from nidaqmx.constants import (AcquisitionType, AccelSensitivityUnits, 
                                  ForceIEPESensorSensitivityUnits, ExcitationSource)
    HAS_NIDAQ = True
except ImportError:
    HAS_NIDAQ = False

try:
    import pyuff
    HAS_PYUFF = True
except ImportError:
    HAS_PYUFF = False

try:
    import pyEMA
    HAS_PYEMA = True
except ImportError:
    HAS_PYEMA = False

def get_disp_unit(unit_str):
    if unit_str == 'mV/g': return 'g'
    if unit_str == 'mV/(m/s²)': return 'm/s²'
    if unit_str == 'mV/N': return 'N'
    if unit_str == 'mV/lbf': return 'lbf'
    return ''

class DummyAcc:
    pass

class AcquisitionThread(QThread):
    data_signal = pyqtSignal(np.ndarray, np.ndarray, bool) 
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def detect_double_hit(self, force_signal, fs):
        if not self.config.get('check_double_hit', True): return False
        max_force = np.max(force_signal)
        if max_force <= 0: return False
        threshold = max_force * 0.20 
        distance_samples = int(fs * 0.01)
        peaks, _ = signal.find_peaks(force_signal, height=threshold, distance=distance_samples)
        return len(peaks) > 1

    def remove_dc_offset(self, data):
        for i in range(data.shape[0]):
            baseline = np.mean(data[i, :])
            data[i, :] -= baseline
        return data

    def apply_units(self, data, channels):
        for idx, ch in enumerate(channels):
            if ch['type'] == 'Aceleração' and ch.get('unit') == 'mV/(m/s²)':
                data[idx, :] *= 9.80665
            elif ch['type'] == 'Força' and ch.get('unit') == 'mV/lbf':
                data[idx, :] /= 4.4482216
        return data

    def run(self):
        if not HAS_NIDAQ:
            self.error_signal.emit("A biblioteca 'nidaqmx' não está instalada ou hardware não detectado.")
            self.finished_signal.emit()
            return
            
        fs = self.config['fs']
        nsamples = self.config['nsamples']
        duration = nsamples / fs
        channels = self.config['channels']
        
        try:
            with nidaqmx.Task() as task:
                hammer_idx_real = -1
                for idx, ch in enumerate(channels):
                    physical_ch = f"{ch['device']}/{ch['channel']}"
                    use_iepe = "Ativado" in ch.get('iepe', 'Ativado')
                    excit_val, excit_src = (0.002, ExcitationSource.INTERNAL) if use_iepe else (0.0, ExcitationSource.NONE)

                    sens_val = ch['sensitivity']
                    
                    if ch['type'] == 'Aceleração':
                        if ch.get('unit') == 'mV/(m/s²)': sens_val = sens_val * 9.80665
                        task.ai_channels.add_ai_accel_chan(
                            physical_channel=physical_ch, min_val=-50.0, max_val=50.0,
                            sensitivity=sens_val, sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,  
                            current_excit_source=excit_src, current_excit_val=excit_val   
                        )
                    elif ch['type'] == 'Força':
                        hammer_idx_real = idx
                        if ch.get('unit') == 'mV/lbf': sens_val = sens_val / 4.4482216
                        task.ai_channels.add_ai_force_iepe_chan(
                            physical_channel=physical_ch, min_val=-200.0, max_val=200.0,
                            sensitivity=sens_val, sensitivity_units=ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,  
                            current_excit_source=excit_src, current_excit_val=excit_val   
                        )
                
                task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
                task.start()
                
                pre_trig_samps = int(fs * 0.05)
                total_samps = nsamples
                chunk_size = int(fs * 0.05)    
                
                buffer = np.zeros((len(channels), 0))
                triggered = False
                data_array = None
                
                while self.is_running:
                    chunk = task.read(number_of_samples_per_channel=chunk_size, timeout=1.0)
                    chunk_np = np.array(chunk)
                    if len(channels) == 1: chunk_np = chunk_np.reshape((1, -1))
                    
                    if not self.config['use_trigger'] or hammer_idx_real == -1:
                        buffer = np.hstack((buffer, chunk_np))
                        if buffer.shape[1] >= total_samps:
                            data_array = buffer[:, :total_samps]
                            break
                        continue
                        
                    hammer_chunk = chunk_np[hammer_idx_real, :]
                    hammer_chunk_ac = hammer_chunk - np.mean(hammer_chunk)
                    
                    trigger_lvl = self.config['trigger_level']
                    if channels[hammer_idx_real].get('unit') == 'mV/lbf': 
                        trigger_lvl = trigger_lvl * 4.4482216
                        
                    if not triggered:
                        if np.max(np.abs(hammer_chunk_ac)) >= trigger_lvl:
                            triggered = True
                            buffer = np.hstack((buffer, chunk_np))
                            
                            samps_to_read = total_samps
                            rest_data = task.read(number_of_samples_per_channel=samps_to_read, timeout=duration + 5.0)
                            rest_np = np.array(rest_data)
                            if len(channels) == 1: rest_np = rest_np.reshape((1, -1))
                            
                            buffer = np.hstack((buffer, rest_np))
                            
                            full_hammer = buffer[hammer_idx_real, :]
                            full_hammer_ac = full_hammer - np.mean(full_hammer)
                            trig_indices = np.where(np.abs(full_hammer_ac) >= trigger_lvl)[0]
                            
                            first_trig = trig_indices[0]
                            start_idx = max(0, first_trig - pre_trig_samps)
                            end_idx = start_idx + total_samps
                            
                            data_array = buffer[:, start_idx:end_idx]
                            if data_array.shape[1] < total_samps:
                                pad_width = total_samps - data_array.shape[1]
                                data_array = np.pad(data_array, ((0,0), (0, pad_width)), mode='edge')
                            break 
                        else:
                            buffer = np.hstack((buffer, chunk_np))
                            if buffer.shape[1] > pre_trig_samps:
                                buffer = buffer[:, -pre_trig_samps:]
                
                task.stop() 
                
                if data_array is None or not self.is_running:
                    self.finished_signal.emit()
                    return
                
                data_array = self.remove_dc_offset(data_array)
                data_array = self.apply_units(data_array, channels)
                t = np.linspace(0, duration, total_samps, endpoint=False)
                
                double_hit = False
                if hammer_idx_real != -1: double_hit = self.detect_double_hit(data_array[hammer_idx_real, :], fs)
                if self.is_running: self.data_signal.emit(t, data_array, double_hit)
                self.finished_signal.emit()
                
        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished_signal.emit()

class EMAVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plataforma de Análise Modal Experimental (EMA)")
        self.resize(1350, 950)
        
        self.hardware_channels = []
        for mod in [1, 2]:
            for ai in range(4):
                is_hammer = (mod == 1 and ai == 0)
                lbl = "Martelo" if is_hammer else f"Acel. Viga - Ponto {ai}"
                self.hardware_channels.append({
                    'device': f"cDAQ1Mod{mod}", 'channel': f"ai{ai}",
                    'enabled': True if (mod == 1 and ai == 0) or (mod == 1 and ai == 1) else False,
                    'type': 'Força' if is_hammer else ('Aceleração' if (mod == 1 and ai == 1) else 'Desativado'),
                    'label': lbl, 'sensitivity': 10.0 if is_hammer else 100.0,
                    'unit': 'mV/N' if is_hammer else 'mV/g', 'iepe': 'Ativado (2mA)' if mod == 1 else 'Desativado',
                    'node': 1 if is_hammer else (ai+1),
                    'dir': '+Z'
                })
                
        self.current_impact = 1
        self.current_node = 1
        self.node_data_buffer = [] 
        self.acq_thread = None
        self.current_project_file = None
        
        self.build_main_app()

    def build_main_app(self):
        self.main_app_widget = QWidget()
        self.setCentralWidget(self.main_app_widget)
        main_layout = QVBoxLayout(self.main_app_widget)
        
        # --- PAINEL GLOBAL ---
        global_frame = QFrame()
        global_frame.setMaximumHeight(60) 
        global_layout = QHBoxLayout(global_frame)
        global_layout.setContentsMargins(5, 5, 5, 5)
        
        # Botões de Projeto (Novo, Carregar, Salvar)
        btn_load_proj = QPushButton("📂 Carregar Projeto")
        btn_load_proj.setStyleSheet("background-color: #6e7781; color: white; font-weight: bold;")
        btn_load_proj.clicked.connect(self.load_project)
        global_layout.addWidget(btn_load_proj)
        
        btn_save_proj = QPushButton("💾 Salvar")
        btn_save_proj.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold;")
        btn_save_proj.clicked.connect(self.save_project)
        global_layout.addWidget(btn_save_proj)
        
        btn_save_as_proj = QPushButton("💾 Salvar Como...")
        btn_save_as_proj.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold;")
        btn_save_as_proj.clicked.connect(self.save_project_as)
        global_layout.addWidget(btn_save_as_proj)
        
        # Separador visual
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        global_layout.addWidget(line)
        
        global_layout.addWidget(QLabel("Diretório:"))
        self.txt_dir = QLineEdit(os.getcwd())
        self.txt_dir.textChanged.connect(self.change_working_directory)
        global_layout.addWidget(self.txt_dir)
        btn_browse = QPushButton("Procurar")
        btn_browse.clicked.connect(self.browse_directory)
        global_layout.addWidget(btn_browse)
        
        self.lbl_status = QLabel("Aguardando verificação...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        global_layout.addWidget(self.lbl_status)
        btn_refresh_hw = QPushButton("🔄 Atualizar Hardware")
        btn_refresh_hw.setStyleSheet("background-color: #d4a72c; font-weight: bold; padding: 5px;")
        btn_refresh_hw.clicked.connect(lambda: self.check_hardware_connection(manual=True))
        global_layout.addWidget(btn_refresh_hw)
        
        main_layout.addWidget(global_frame)

        self.tabs = QTabWidget()
        self.tab_geom = QWidget(); self.setupTab_Geom()
        self.tab_ch = QWidget(); self.setupTab_Channels()
        self.tab_acq = QWidget(); self.setupTab_Acquisition()
        self.tab_ema = QWidget(); self.setupTab_pyEMA()
        
        self.tabs.addTab(self.tab_geom, "1. Geometria 3D")
        self.tabs.addTab(self.tab_ch, "2. Mapeamento de Canais")
        self.tabs.addTab(self.tab_acq, "3. Aquisição Modal (Martelo)")
        self.tabs.addTab(self.tab_ema, "4. Análise Modal (pyEMA)")
        
        main_layout.addWidget(self.tabs)
        self.check_hardware_connection(manual=False)

    # ===============================================================
    # GERENCIAMENTO DE PROJETO (SALVAR / CARREGAR)
    # ===============================================================
    def save_project(self):
        if self.current_project_file is None:
            self.save_project_as()
        else:
            self._write_project_file(self.current_project_file)
            
    def save_project_as(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Salvar Projeto Como", "Projeto_EMA.json", "Arquivos de Projeto JSON (*.json);;Todos os Arquivos (*)", options=options)
        if filename:
            self.current_project_file = filename
            self._write_project_file(filename)

    def _write_project_file(self, filename):
        proj_data = {}
        
        # 1. Geometria
        geom_nodes = []
        for r in range(self.table_nodes.rowCount()):
            try:
                geom_nodes.append({
                    'id': self.table_nodes.item(r, 0).text(),
                    'x': self.table_nodes.item(r, 1).text(),
                    'y': self.table_nodes.item(r, 2).text(),
                    'z': self.table_nodes.item(r, 3).text()
                })
            except: pass
        geom_lines = []
        for r in range(self.table_lines.rowCount()):
            try:
                geom_lines.append({
                    'n1': self.table_lines.item(r, 0).text(),
                    'n2': self.table_lines.item(r, 1).text()
                })
            except: pass
        proj_data['geometry'] = {'nodes': geom_nodes, 'lines': geom_lines}

        # 2. Canais
        ch_data = []
        for i in range(self.table_instruments.rowCount()):
            parts = self.table_instruments.cellWidget(i, 1).text().split('/')
            ch_data.append({
                'enabled': self.table_instruments.cellWidget(i, 0).isChecked(),
                'device': parts[0], 'channel': parts[1] if len(parts) > 1 else 'ai0',
                'type': self.table_instruments.cellWidget(i, 2).currentText(),
                'node': self.table_instruments.cellWidget(i, 3).value(),
                'dir': self.table_instruments.cellWidget(i, 4).currentText(),
                'sensitivity': self.table_instruments.cellWidget(i, 5).value(),
                'unit': self.table_instruments.cellWidget(i, 6).currentText(),
                'iepe': self.table_instruments.cellWidget(i, 7).currentText()
            })
        proj_data['channels'] = ch_data
        
        # 3. Arquivos UFF importados
        uffs = [self.list_uffs.item(i, 0).text() for i in range(self.list_uffs.rowCount())]
        proj_data['uffs'] = uffs
        
        # 4. Parâmetros da EMA
        proj_data['ema_params'] = {
            'fs': self.c_fs_ema.currentText(),
            'pts': self.c_pts_ema.currentText(),
            'pretrig': self.s_pretrig.value(),
            'fmin': self.sp_fmin.value(),
            'fmax': self.sp_fmax.value(),
            'window': self.c_window.currentText(),
            'map_base': self.c_map_base.currentText()
        }
        
        # 5. Resultados (Separa a parte Real e Imaginária para não quebrar o JSON)
        if hasattr(self, 'A_shapes') and hasattr(self, 'acc') and len(self.acc.nat_freq) > 0:
            proj_data['results'] = {
                'has_results': True,
                'nat_freq': self.acc.nat_freq.tolist() if isinstance(self.acc.nat_freq, np.ndarray) else list(self.acc.nat_freq),
                'nat_xi': self.acc.nat_xi.tolist() if isinstance(self.acc.nat_xi, np.ndarray) else list(self.acc.nat_xi),
                'frf_dofs': self.frf_dofs,
                'A_shapes_real': np.real(self.A_shapes).tolist(),
                'A_shapes_imag': np.imag(self.A_shapes).tolist()
            }
        else:
            proj_data['results'] = {'has_results': False}
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(proj_data, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Projeto Salvo", f"Projeto salvo com sucesso em:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Salvar", f"Ocorreu um erro ao salvar o projeto:\n{str(e)}")

    def load_project(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Carregar Projeto", "", "Arquivos de Projeto JSON (*.json);;Todos os Arquivos (*)", options=options)
        if not filename: return
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                proj_data = json.load(f)
            
            self.current_project_file = filename
            
            # 1. Carrega Geometria
            if 'geometry' in proj_data:
                geom = proj_data['geometry']
                self.table_nodes.setRowCount(0)
                for node in geom.get('nodes', []):
                    row = self.table_nodes.rowCount(); self.table_nodes.insertRow(row)
                    self.table_nodes.setItem(row, 0, QTableWidgetItem(str(node.get('id', ''))))
                    self.table_nodes.setItem(row, 1, QTableWidgetItem(str(node.get('x', '0'))))
                    self.table_nodes.setItem(row, 2, QTableWidgetItem(str(node.get('y', '0'))))
                    self.table_nodes.setItem(row, 3, QTableWidgetItem(str(node.get('z', '0'))))
                
                self.table_lines.setRowCount(0)
                for line in geom.get('lines', []):
                    row = self.table_lines.rowCount(); self.table_lines.insertRow(row)
                    self.table_lines.setItem(row, 0, QTableWidgetItem(str(line.get('n1', ''))))
                    self.table_lines.setItem(row, 1, QTableWidgetItem(str(line.get('n2', ''))))
                self.plot_geometry()
                
            # 2. Carrega Canais
            if 'channels' in proj_data:
                self.table_instruments.setRowCount(0)
                for ch in proj_data['channels']:
                    self.add_instrument_row(ch)
                    
            # 3. Carrega UFFs
            if 'uffs' in proj_data:
                self.list_uffs.setRowCount(0)
                for uff_file in proj_data['uffs']:
                    r = self.list_uffs.rowCount(); self.list_uffs.insertRow(r); self.list_uffs.setItem(r, 0, QTableWidgetItem(uff_file))
                    
            # 4. Carrega Parâmetros
            if 'ema_params' in proj_data:
                params = proj_data['ema_params']
                self.c_fs_ema.setCurrentText(str(params.get('fs', '4096')))
                self.c_pts_ema.setCurrentText(str(params.get('pts', '2048')))
                self.s_pretrig.setValue(params.get('pretrig', 1))
                self.sp_fmin.setValue(params.get('fmin', 10))
                self.sp_fmax.setValue(params.get('fmax', 600))
                self.c_window.setCurrentText(params.get('window', 'Retangular (Nenhum)'))
                self.c_map_base.setCurrentText(params.get('map_base', 'Auto (Detectar pelo arquivo UFF)'))
                
            # 5. Reconstrói Resultados
            if 'results' in proj_data and proj_data['results'].get('has_results', False):
                res = proj_data['results']
                self.acc = DummyAcc()
                self.acc.nat_freq = np.array(res['nat_freq'])
                self.acc.nat_xi = np.array(res['nat_xi'])
                self.frf_dofs = res['frf_dofs']
                
                real_part = np.array(res['A_shapes_real'])
                imag_part = np.array(res['A_shapes_imag'])
                self.A_shapes = real_part + 1j * imag_part
                
                self.c_view_mode.blockSignals(True)
                self.c_view_mode.clear()
                for i, (fn, zeta) in enumerate(zip(self.acc.nat_freq, self.acc.nat_xi)):
                    self.c_view_mode.addItem(f"Modo {i+1}: {fn:.2f} Hz, amort: {zeta*100:.2f}%")
                self.c_view_mode.blockSignals(False)
                
                self.btn_save_res.setEnabled(True)
                self.plot_3d_mode()
            else:
                self.clear_analysis_data()
                
            self.setWindowTitle(f"Plataforma de Análise Modal Experimental (EMA) - {os.path.basename(filename)}")
            QMessageBox.information(self, "Projeto Carregado", "O Projeto foi carregado com sucesso!")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Carregar", f"Falha ao carregar o arquivo de projeto:\n{str(e)}")

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Selecionar Diretório", self.txt_dir.text())
        if dir_path: self.txt_dir.setText(dir_path)

    def change_working_directory(self, path):
        if os.path.isdir(path):
            try: os.chdir(path)
            except Exception: pass

    def check_hardware_connection(self, manual=False):
        connected = False
        msg = "❌ Hardware Não Detectado Fisicamente"
        if HAS_NIDAQ:
            try:
                system = nidaqmx.system.System.local()
                active_devs = []
                for dev in system.devices:
                    if not dev.is_simulated:
                        active_devs.append(dev.name)
                if active_devs:
                    msg = f"✅ Hardware OK: {', '.join(active_devs)}"
                    connected = True
            except: pass
            
        if connected:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("background-color: #2da44e; color: white; padding: 5px; font-weight: bold;")
            if manual: QMessageBox.information(self, "Status", msg)
        else:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("background-color: #cf222e; color: white; padding: 5px; font-weight: bold;")
            if manual: QMessageBox.warning(self, "Status", msg)

    # ABA 1: GEOMETRIA
    def setupTab_Geom(self):
        layout = QHBoxLayout(self.tab_geom)
        left_panel = QVBoxLayout()
        
        group_nodes = QGroupBox("Nós (Coordenadas mm)")
        lay_nodes = QVBoxLayout()
        self.table_nodes = QTableWidget(5, 4)
        self.table_nodes.setHorizontalHeaderLabels(["ID do Nó", "X", "Y", "Z"])
        self.table_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        default_nodes = [(1,0,0,0), (2,74.5,0,0), (3,149,0,0), (4,223.5,0,0), (5,298,0,0)]
        for i, (nid, x, y, z) in enumerate(default_nodes):
            self.table_nodes.setItem(i, 0, QTableWidgetItem(str(nid)))
            self.table_nodes.setItem(i, 1, QTableWidgetItem(str(x)))
            self.table_nodes.setItem(i, 2, QTableWidgetItem(str(y)))
            self.table_nodes.setItem(i, 3, QTableWidgetItem(str(z)))
            
        node_btn_layout = QHBoxLayout()
        btn_add_node = QPushButton("+ Adicionar Nó"); btn_add_node.clicked.connect(self.add_node_row)
        btn_del_node = QPushButton("- Excluir Nó"); btn_del_node.clicked.connect(self.delete_node_row)
        node_btn_layout.addWidget(btn_add_node); node_btn_layout.addWidget(btn_del_node)

        lay_nodes.addWidget(self.table_nodes)
        lay_nodes.addLayout(node_btn_layout)
        group_nodes.setLayout(lay_nodes)
        left_panel.addWidget(group_nodes)

        group_lines = QGroupBox("Linhas (Wireframe)")
        lay_lines = QVBoxLayout()
        self.table_lines = QTableWidget(4, 2)
        self.table_lines.setHorizontalHeaderLabels(["Nó Origem", "Nó Destino"])
        self.table_lines.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        default_lines = [(1,2), (2,3), (3,4), (4,5)]
        for i, (n1, n2) in enumerate(default_lines):
            self.table_lines.setItem(i, 0, QTableWidgetItem(str(n1)))
            self.table_lines.setItem(i, 1, QTableWidgetItem(str(n2)))
            
        line_btn_layout = QHBoxLayout()
        btn_add_line = QPushButton("+ Adicionar Linha"); btn_add_line.clicked.connect(self.add_line_row)
        btn_del_line = QPushButton("- Excluir Linha"); btn_del_line.clicked.connect(self.delete_line_row)
        line_btn_layout.addWidget(btn_add_line); line_btn_layout.addWidget(btn_del_line)

        lay_lines.addWidget(self.table_lines)
        lay_lines.addLayout(line_btn_layout)
        group_lines.setLayout(lay_lines)
        left_panel.addWidget(group_lines)
        
        btn_update = QPushButton("Atualizar Gráfico 3D")
        btn_update.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold; min-height: 40px;")
        btn_update.clicked.connect(self.plot_geometry)
        left_panel.addWidget(btn_update)
        
        right_panel = QVBoxLayout()
        self.fig_geom = Figure()
        self.canvas_geom = FigureCanvas(self.fig_geom)
        right_panel.addWidget(self.canvas_geom)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)
        QTimer.singleShot(500, self.plot_geometry)

    def add_node_row(self):
        row = self.table_nodes.rowCount(); self.table_nodes.insertRow(row)
        self.table_nodes.setItem(row, 0, QTableWidgetItem(str(row + 1))); self.table_nodes.setItem(row, 1, QTableWidgetItem("0.0")); self.table_nodes.setItem(row, 2, QTableWidgetItem("0.0")); self.table_nodes.setItem(row, 3, QTableWidgetItem("0.0"))
        self.plot_geometry()

    def delete_node_row(self):
        selected = set(item.row() for item in self.table_nodes.selectedItems())
        if selected:
            for row in sorted(selected, reverse=True): self.table_nodes.removeRow(row)
        elif self.table_nodes.rowCount() > 0: self.table_nodes.removeRow(self.table_nodes.rowCount() - 1)
        self.plot_geometry()

    def add_line_row(self):
        row = self.table_lines.rowCount(); self.table_lines.insertRow(row)
        self.table_lines.setItem(row, 0, QTableWidgetItem("1")); self.table_lines.setItem(row, 1, QTableWidgetItem("2"))
        self.plot_geometry()

    def delete_line_row(self):
        selected = set(item.row() for item in self.table_lines.selectedItems())
        if selected:
            for row in sorted(selected, reverse=True): self.table_lines.removeRow(row)
        elif self.table_lines.rowCount() > 0: self.table_lines.removeRow(self.table_lines.rowCount() - 1)
        self.plot_geometry()

    def get_geometry_data(self):
        nodes = {}
        for r in range(self.table_nodes.rowCount()):
            try:
                nid = int(self.table_nodes.item(r, 0).text())
                x = float(self.table_nodes.item(r, 1).text())
                y = float(self.table_nodes.item(r, 2).text())
                z = float(self.table_nodes.item(r, 3).text())
                nodes[nid] = (x, y, z)
            except: pass
        lines = []
        for r in range(self.table_lines.rowCount()):
            try:
                n1 = int(self.table_lines.item(r, 0).text())
                n2 = int(self.table_lines.item(r, 1).text())
                lines.append((n1, n2))
            except: pass
        return nodes, lines

    def plot_geometry(self):
        nodes, lines = self.get_geometry_data()
        self.fig_geom.clf()
        ax = self.fig_geom.add_subplot(111, projection='3d')
        
        for n1, n2 in lines:
            if n1 in nodes and n2 in nodes:
                p1, p2 = nodes[n1], nodes[n2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', lw=3, alpha=0.8)
                
        for nid, (x, y, z) in nodes.items():
            ax.scatter(x, y, z, c='r', s=60, edgecolors='black', zorder=5)
            ax.text(x, y, z, f'  Nó {nid}', color='black', fontsize=10, fontweight='bold', zorder=10)
            
        ax.set_xlabel('X (mm)', fontweight='bold'); ax.set_ylabel('Y (mm)', fontweight='bold'); ax.set_zlabel('Z (mm)', fontweight='bold')
        ax.set_title("Modelo Geométrico da Estrutura", pad=15, fontsize=12, fontweight='bold')
        
        if nodes:
            coords = np.array(list(nodes.values()))
            x_min, x_max = coords[:,0].min(), coords[:,0].max()
            y_min, y_max = coords[:,1].min(), coords[:,1].max()
            z_min, z_max = coords[:,2].min(), coords[:,2].max()
            dx = max(x_max - x_min, 1.0); dy = max(y_max - y_min, 1.0); dz = max(z_max - z_min, 1.0)
            ax.set_xlim(x_min - dx*0.1, x_max + dx*0.1)
            ax.set_ylim(y_min - dy*0.1, y_max + dy*0.1)
            ax.set_zlim(z_min - dz*0.1, z_max + dz*0.1)
            try: ax.set_box_aspect((dx, max(dy, max(dx,dy,dz)*0.3), max(dz, max(dx,dy,dz)*0.3)))
            except AttributeError: pass
        ax.view_init(elev=20, azim=-60)
        self.fig_geom.tight_layout()
        self.canvas_geom.draw()

    # ABA 2: CANAIS
    def setupTab_Channels(self):
        layout = QVBoxLayout(self.tab_ch)
        self.table_instruments = QTableWidget(0, 8)
        self.table_instruments.setHorizontalHeaderLabels(["Ativar", "Dispositivo/Canal", "Tipo", "Nó Físico", "Direção", "Sensibilidade", "Unidade", "IEPE"])
        self.table_instruments.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_instruments)
        
        ch_btn_layout = QHBoxLayout()
        btn_add_ch = QPushButton("+ Adicionar Canal"); btn_add_ch.clicked.connect(lambda: self.add_instrument_row())
        btn_del_ch = QPushButton("- Remover Canal"); btn_del_ch.clicked.connect(self.delete_instrument_row)
        ch_btn_layout.addWidget(btn_add_ch); ch_btn_layout.addWidget(btn_del_ch)
        layout.addLayout(ch_btn_layout)
        
        for ch in self.hardware_channels[:2]:
            self.add_instrument_row(ch)
            
        btn_next = QPushButton("Avançar >>")
        btn_next.clicked.connect(lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1))
        layout.addWidget(btn_next)

    def add_instrument_row(self, ch=None):
        row = self.table_instruments.rowCount()
        self.table_instruments.insertRow(row)
        if ch is None: ch = {'device': 'cDAQ1Mod1', 'channel': f'ai{row}', 'enabled': True, 'type': 'Aceleração', 'node': 1, 'dir': '+Z', 'sensitivity': 100.0, 'unit': 'mV/g', 'iepe': 'Ativado (2mA)'}
            
        chk = QCheckBox(); chk.setChecked(ch.get('enabled', True))
        self.table_instruments.setCellWidget(row, 0, chk)
        self.table_instruments.setCellWidget(row, 1, QLineEdit(f"{ch.get('device', 'cDAQ1Mod1')}/{ch.get('channel', 'ai0')}"))
        
        combo_type = QComboBox(); combo_type.addItems(["Desativado", "Aceleração", "Força"])
        combo_type.setCurrentText(ch.get('type', 'Desativado'))
        combo_type.currentTextChanged.connect(self._on_type_changed)
        self.table_instruments.setCellWidget(row, 2, combo_type)
        
        spin_node = QSpinBox(); spin_node.setRange(1, 9999); spin_node.setValue(ch.get('node', 1))
        self.table_instruments.setCellWidget(row, 3, spin_node)
        
        combo_dir = QComboBox(); combo_dir.addItems(["+X", "-X", "+Y", "-Y", "+Z", "-Z"]); combo_dir.setCurrentText(ch.get('dir', '+Z'))
        self.table_instruments.setCellWidget(row, 4, combo_dir)
        
        spin = QDoubleSpinBox(); spin.setRange(0.001, 10000.0); spin.setValue(ch.get('sensitivity', 100.0)); spin.setDecimals(3)
        self.table_instruments.setCellWidget(row, 5, spin)
        
        combo_unit = QComboBox()
        self.table_instruments.setCellWidget(row, 6, combo_unit)
        
        combo_iepe = QComboBox(); combo_iepe.addItems(["Desativado", "Ativado (2mA)"]); combo_iepe.setCurrentText(ch.get('iepe', 'Desativado'))
        self.table_instruments.setCellWidget(row, 7, combo_iepe)
        self._update_sens_unit(row, combo_type.currentText())
        combo_unit.setCurrentText(ch.get('unit', 'mV/g' if ch.get('type') == 'Aceleração' else 'mV/N'))

    def delete_instrument_row(self):
        rows = set(item.row() for item in self.table_instruments.selectedItems())
        if not rows and self.table_instruments.rowCount() > 0: self.table_instruments.removeRow(self.table_instruments.rowCount() - 1)
        else:
            for row in sorted(rows, reverse=True): self.table_instruments.removeRow(row)

    def _on_type_changed(self, text):
        sender = self.sender()
        for i in range(self.table_instruments.rowCount()):
            if self.table_instruments.cellWidget(i, 2) == sender:
                self._update_sens_unit(i, text); break

    def _update_sens_unit(self, row, text):
        spin = self.table_instruments.cellWidget(row, 5)
        combo_unit = self.table_instruments.cellWidget(row, 6)
        
        current_unit = combo_unit.currentText()
        combo_unit.blockSignals(True)
        combo_unit.clear()
        spin.setSuffix("")
        
        if text == "Aceleração": 
            combo_unit.addItems(["mV/g", "mV/(m/s²)"])
            if current_unit in ["mV/g", "mV/(m/s²)"]: combo_unit.setCurrentText(current_unit)
        elif text == "Força": 
            combo_unit.addItems(["mV/N", "mV/lbf"])
            if current_unit in ["mV/N", "mV/lbf"]: combo_unit.setCurrentText(current_unit)
        else: 
            combo_unit.addItems(["-"])
            
        combo_unit.blockSignals(False)

    def get_active_channels(self):
        active = []
        for i in range(self.table_instruments.rowCount()):
            chk = self.table_instruments.cellWidget(i, 0)
            combo = self.table_instruments.cellWidget(i, 2)
            if chk and chk.isChecked() and combo.currentText() != "Desativado":
                parts = self.table_instruments.cellWidget(i, 1).text().split('/')
                node_val = self.table_instruments.cellWidget(i, 3).value()
                dir_val = self.table_instruments.cellWidget(i, 4).currentText()
                tipo = combo.currentText()
                lbl = "Martelo" if tipo == 'Força' else f"Nó {node_val} ({dir_val})"
                active.append({
                    'device': parts[0], 'channel': parts[1] if len(parts) > 1 else 'ai0', 
                    'type': tipo, 'label': lbl, 'sensitivity': self.table_instruments.cellWidget(i, 5).value(),
                    'unit': self.table_instruments.cellWidget(i, 6).currentText(),
                    'node': node_val, 'dir': dir_val, 'iepe': self.table_instruments.cellWidget(i, 7).currentText()
                })
        return active

    # ABA 3: AQUISIÇÃO (EMA)
    def setupTab_Acquisition(self):
        main_layout = QHBoxLayout(self.tab_acq)
        left_panel = QVBoxLayout()
        
        group_timing = QGroupBox("Parâmetros de Amostragem")
        grid_timing = QGridLayout()
        grid_timing.addWidget(QLabel("Taxa de Amostragem (Hz):"), 0, 0)
        self.c_fs_acq = QComboBox()
        self.c_fs_acq.addItems(["1024", "2048", "4096", "8192", "16384"])
        self.c_fs_acq.setEditable(True); self.c_fs_acq.setCurrentText("4096")
        grid_timing.addWidget(self.c_fs_acq, 0, 1)
        
        grid_timing.addWidget(QLabel("Número de Pontos:"), 1, 0)
        self.c_pts_acq = QComboBox()
        self.c_pts_acq.addItems(["512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"])
        self.c_pts_acq.setEditable(True); self.c_pts_acq.setCurrentText("8192")
        grid_timing.addWidget(self.c_pts_acq, 1, 1)

        self.lbl_timing_acq = QLabel()
        grid_timing.addWidget(self.lbl_timing_acq, 2, 0, 1, 2)
        group_timing.setLayout(grid_timing)
        left_panel.addWidget(group_timing)

        self.c_fs_acq.currentTextChanged.connect(self.update_timing_acq)
        self.c_pts_acq.currentTextChanged.connect(self.update_timing_acq)
        self.update_timing_acq()
        
        self.txt_export_name = QLineEdit("Teste_Viga")
        left_panel.addWidget(QLabel("Prefixo do Arquivo de Exportação (.uff):"))
        left_panel.addWidget(self.txt_export_name)
        
        self.group_trigger = QGroupBox("Trigger e Qualidade (Apenas EMA)")
        grid_trig = QGridLayout()
        grid_trig.addWidget(QLabel("Nível Trigger (N):"), 0, 0)
        self.spin_trig_lvl = QDoubleSpinBox()
        self.spin_trig_lvl.setRange(0.1, 1000.0); self.spin_trig_lvl.setValue(5.0)
        grid_trig.addWidget(self.spin_trig_lvl, 0, 1)
        self.chk_trigger = QCheckBox("Habilitar Gatilho (Trigger) no Martelo")
        self.chk_trigger.setChecked(True)
        grid_trig.addWidget(self.chk_trigger, 1, 0, 1, 2)
        self.chk_double_hit = QCheckBox("Detectar e Alertar Dupla Batida")
        self.chk_double_hit.setChecked(True)
        grid_trig.addWidget(self.chk_double_hit, 2, 0, 1, 2)
        self.group_trigger.setLayout(grid_trig)
        left_panel.addWidget(self.group_trigger)
        
        self.group_modal = QGroupBox("Controle de Malha de Teste")
        grid_modal = QGridLayout()
        grid_modal.addWidget(QLabel("O que você está movendo?"), 0, 0, 1, 2)
        self.c_roving = QComboBox(); self.c_roving.addItems(["Martelo Móvel (Roving Hammer)", "Acelerômetro Móvel (Roving Accel)"])
        grid_modal.addWidget(self.c_roving, 1, 0, 1, 2)
        grid_modal.addWidget(QLabel("Qtd Total de Nós:"), 2, 0)
        self.spin_nodes = QSpinBox(); self.spin_nodes.setValue(5); self.spin_nodes.setMinimum(1)
        grid_modal.addWidget(self.spin_nodes, 2, 1)
        grid_modal.addWidget(QLabel("Nó Atual:"), 3, 0)
        self.s_curr_node = QSpinBox(); self.s_curr_node.setRange(1, 9999); self.s_curr_node.setValue(1)
        grid_modal.addWidget(self.s_curr_node, 3, 1)
        grid_modal.addWidget(QLabel("Médias/Impactos por Nó:"), 4, 0)
        self.spin_impacts = QSpinBox(); self.spin_impacts.setValue(3); self.spin_impacts.setMinimum(1)
        grid_modal.addWidget(self.spin_impacts, 4, 1)
        self.lbl_progress = QLabel("Nó: 1/5 | Impacto: 1/3")
        self.lbl_progress.setStyleSheet("font-weight: bold; color: blue;")
        grid_modal.addWidget(self.lbl_progress, 5, 0, 1, 2, Qt.AlignCenter)
        self.group_modal.setLayout(grid_modal)
        left_panel.addWidget(self.group_modal)
        
        self.btn_start_ema = QPushButton("Armar Aquisição (Aguardar Impacto)")
        self.btn_start_ema.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_start_ema.clicked.connect(self.start_ema_acquisition)
        left_panel.addWidget(self.btn_start_ema)
        
        self.btn_reject_ema = QPushButton("Rejeitar Último Impacto")
        self.btn_reject_ema.setStyleSheet("background-color: #d73a49; color: white; min-height: 30px; font-weight: bold;")
        self.btn_reject_ema.setEnabled(False)
        self.btn_reject_ema.clicked.connect(self.reject_last_impact)
        left_panel.addWidget(self.btn_reject_ema)
        
        self.btn_stop = QPushButton("Parar Execução")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_acquisition)
        left_panel.addWidget(self.btn_stop)
        left_panel.addStretch()
        
        right_panel = QVBoxLayout()
        self.figure = Figure()
        self.canvas_sig = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas_sig)
        
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.ax1.set_title("Sinal de Força (Martelo)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True)
        self.ax2.set_title("Resposta Vibratória (Acelerômetros)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.set_xlabel("Tempo [s]")
        self.ax2.grid(True)
        self.figure.tight_layout()
        
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)

    def update_timing_acq(self):
        try:
            fs = float(self.c_fs_acq.currentText())
            pts = int(self.c_pts_acq.currentText())
            dur = pts / fs; df = fs / pts
            self.lbl_timing_acq.setText(f"Duração: {dur:.3f} s | Resolução (Δf): {df:.3f} Hz")
            self.lbl_timing_acq.setStyleSheet("font-weight: bold; color: #0366d6;")
        except ValueError:
            self.lbl_timing_acq.setText("Valores Inválidos!")

    def start_ema_acquisition(self):
        active_channels = self.get_active_channels()
        if not active_channels: return QMessageBox.warning(self, "Erro", "Nenhum canal ativo selecionado!")
        config = {
            'fs': float(self.c_fs_acq.currentText()), 'nsamples': int(self.c_pts_acq.currentText()),
            'channels': active_channels, 'acq_type': 'ema',
            'use_trigger': self.chk_trigger.isChecked(), 'trigger_level': self.spin_trig_lvl.value(),
            'check_double_hit': self.chk_double_hit.isChecked()
        }
        self.btn_start_ema.setText("Aguardando Impacto...")
        self.btn_start_ema.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        if len(self.node_data_buffer) > 0: self.btn_reject_ema.setEnabled(True)
        
        self.acq_thread = AcquisitionThread(config)
        self.acq_thread.data_signal.connect(self.route_data)
        self.acq_thread.error_signal.connect(self.handle_error)
        self.acq_thread.finished_signal.connect(self.acquisition_finished)
        self.acq_thread.start()

    def stop_acquisition(self):
        if hasattr(self, 'acq_thread') and self.acq_thread.isRunning():
            self.acq_thread.is_running = False 
            self.acq_thread.wait(2000)         
        self.acquisition_finished()

    def reject_last_impact(self):
        if not self.node_data_buffer: return
        self.node_data_buffer.pop() 
        self.current_impact = max(1, self.current_impact - 1)    
        self.lbl_progress.setText(f"Nó: {self.current_node}/{self.spin_nodes.value()} | Impacto: {self.current_impact}/{self.spin_impacts.value()}")
        
        self.ax1.clear(); self.ax2.clear()
        self.ax1.set_title("Sinal de Força (Martelo)"); self.ax1.grid(True)
        self.ax2.set_title("Resposta Vibratória"); self.ax2.set_xlabel("Tempo [s]"); self.ax2.grid(True)
        self.ax1.text(0.5, 0.5, 'IMPACTO REJEITADO', transform=self.ax1.transAxes, color='red', fontsize=14, ha='center', weight='bold')
        self.canvas_sig.draw()
        if len(self.node_data_buffer) == 0: self.btn_reject_ema.setEnabled(False)

    def handle_error(self, err_msg):
        QMessageBox.critical(self, "Erro na Placa", f"Ocorreu um erro de comunicação:\n\n{err_msg}")
        self.acquisition_finished()

    def acquisition_finished(self):
        self.btn_start_ema.setText("Armar Aquisição (Aguardar Impacto)")
        self.btn_start_ema.setEnabled(True)
        if len(self.node_data_buffer) > 0: self.btn_reject_ema.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def route_data(self, time_axis, data_matrix, double_hit):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.set_title("Sinal de Força (Martelo)"); self.ax1.grid(True)
        self.ax2.set_title("Resposta Vibratória"); self.ax2.set_xlabel("Tempo [s]"); self.ax2.grid(True)
        
        active = self.get_active_channels()
        for idx, ch in enumerate(active):
            if idx >= data_matrix.shape[0]: break
            lbl_unit = f"{ch['label']} [{get_disp_unit(ch.get('unit', ''))}]"
            if ch['type'] == 'Força': self.ax1.plot(time_axis, data_matrix[idx, :], label=lbl_unit, color='red')
            elif ch['type'] == 'Aceleração': self.ax2.plot(time_axis, data_matrix[idx, :], alpha=0.8, label=lbl_unit)
                
        self.ax1.legend(loc="upper right"); self.ax2.legend(loc="upper right")
        
        if double_hit:
            self.ax1.text(0.5, 0.9, 'DUPLA BATIDA', transform=self.ax1.transAxes, color='white', backgroundcolor='red', fontsize=12, ha='center')
            self.canvas_sig.draw()
            if QMessageBox.question(self, 'Dupla Batida', 'Foi detectada uma dupla batida. REJEITAR este impacto?', QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                QTimer.singleShot(1500, self.start_ema_acquisition)
                return 
        else:
            self.canvas_sig.draw()
            
        self.node_data_buffer.append(data_matrix.copy())
        self.btn_reject_ema.setEnabled(True) 
        
        if self.current_impact < self.spin_impacts.value():
            self.current_impact += 1
            self.lbl_progress.setText(f"Nó: {self.current_node}/{self.spin_nodes.value()} | Impacto: {self.current_impact}/{self.spin_impacts.value()}")
            QTimer.singleShot(1500, self.start_ema_acquisition)
        else:
            self.export_node_to_uff(active)
            self.btn_reject_ema.setEnabled(False) 
            
            if self.current_node < self.spin_nodes.value():
                QMessageBox.information(self, "Troca", f"Média do Nó {self.current_node} concluída.\nMova para o Nó {self.current_node + 1}.")
                self.current_node += 1
                self.current_impact = 1
                self.s_curr_node.setValue(self.current_node)
                self.lbl_progress.setText(f"Nó: {self.current_node}/{self.spin_nodes.value()} | Impacto: {self.current_impact}/{self.spin_impacts.value()}")
            else:
                QMessageBox.information(self, "Fim", "Malha de testes concluída!")
                self.current_node = 1
                self.current_impact = 1
                self.s_curr_node.setValue(self.current_node)
                self.lbl_progress.setText(f"Nó: 1/{self.spin_nodes.value()} | Impacto: 1/{self.spin_impacts.value()}")

    def export_node_to_uff(self, active_channels):
        if not HAS_PYUFF: return
        continuous_data = np.concatenate(self.node_data_buffer, axis=1)
        self.node_data_buffer.clear()
        
        fs = float(self.c_fs_acq.currentText()); dt = 1.0 / fs
        timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        data_formatada = datetime.datetime.now().strftime("%d-%b-%y %H:%M:%S")
        
        filename = f"{self.txt_export_name.text()}_Node{self.current_node}_{timestamp}.uff"
        accels_data = []; force_data = None
        
        for idx, ch in enumerate(active_channels):
            if ch['type'] == 'Aceleração': 
                data_to_export = continuous_data[idx, :]
                if ch.get('unit') == 'mV/g': data_to_export = data_to_export * 9.80665
                accels_data.append((data_to_export, ch))
            elif ch['type'] == 'Força': 
                data_to_export = continuous_data[idx, :]
                if ch.get('unit') == 'mV/lbf': data_to_export = data_to_export * 4.4482216
                force_data = data_to_export
                
        datasets_to_write = []
        for i, (acc_data, ch_info) in enumerate(accels_data):
            dir_map = {'+X': 1, '-X': -1, '+Y': 2, '-Y': -2, '+Z': 3, '-Z': -3}
            ds = {
                'type': 58, 'func_type': 1, 
                'rsp_node': ch_info['node'], 'rsp_dir': dir_map.get(ch_info['dir'], 3), 
                'ref_node': self.current_node, 'ref_dir': 3, 
                'id1': f"Channel {i+1}", 'id2': f'Test Setup: Node {self.current_node}', 
                'id3': data_formatada, 'id4': 'EMA Measurement', 'id5': 'NONE',
                'abscissa_spacing': 1, 'abscissa_spec_data_type': 17, 
                'ordinate_spec_data_type': 12, 'orddenom_spec_data_type': 0, 'z_axis_spec_data_type': 0,
                'abscissa_ident': 'Time', 'abscissa_unit_lbl': 's',
                'ordinate_ident': 'Acceleration', 'ordinate_unit_lbl': 'm/s^2',
                'abscissa_min': 0.0, 'abscissa_inc': dt, 
                'x': np.arange(len(acc_data)) * dt, 'data': acc_data
            }
            datasets_to_write.append(ds)
            
        if force_data is not None:
            ds = {
                'type': 58, 'func_type': 1, 
                'rsp_node': self.current_node, 'rsp_dir': 3, 
                'ref_node': self.current_node, 'ref_dir': 3, 
                'id1': f"Force Channel", 'id2': f'Test Setup: Node {self.current_node}', 
                'id3': data_formatada, 'id4': 'EMA Measurement', 'id5': 'NONE',
                'abscissa_spacing': 1, 'abscissa_spec_data_type': 17, 
                'ordinate_spec_data_type': 13, 'orddenom_spec_data_type': 0, 'z_axis_spec_data_type': 0, 
                'abscissa_ident': 'Time', 'abscissa_unit_lbl': 's',
                'ordinate_ident': 'Force', 'ordinate_unit_lbl': 'N',
                'abscissa_min': 0.0, 'abscissa_inc': dt, 
                'x': np.arange(len(force_data)) * dt, 'data': force_data
            }
            datasets_to_write.append(ds)
            
        try:
            uff_file = pyuff.UFF(filename)
            for ds in datasets_to_write:
                if 'x' in ds:
                    try: uff_file._write_set(ds) 
                    except: pass
            uff_file.write_sets(datasets_to_write, mode='add')
        except Exception as e: QMessageBox.critical(self, "Erro", f"Erro UFF:\n{str(e)}")

    # ABA 4: PYEMA
    def setupTab_pyEMA(self):
        layout = QHBoxLayout(self.tab_ema)
        left_panel = QVBoxLayout()
        
        self.list_uffs = QTableWidget(0, 1)
        self.list_uffs.setHorizontalHeaderLabels(["Arquivos .uff"])
        self.list_uffs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left_panel.addWidget(self.list_uffs)
        
        btn_add = QPushButton("+ Adicionar UFFs"); btn_add.clicked.connect(self.load_uffs)
        left_panel.addWidget(btn_add)
        btn_clear_analysis = QPushButton("Limpar Dados / Resetar"); btn_clear_analysis.setStyleSheet("background-color: #d73a49; color: white; font-weight: bold;"); btn_clear_analysis.clicked.connect(self.clear_analysis_data)
        left_panel.addWidget(btn_clear_analysis)
        
        g_param = QGroupBox("Parâmetros de Cálculo da FRF (H1)")
        gp = QGridLayout()
        gp.addWidget(QLabel("Taxa de Amostragem (Hz):"), 0, 0)
        self.c_fs_ema = QComboBox(); self.c_fs_ema.addItems(["1024", "2048", "4096", "8192", "16384"]); self.c_fs_ema.setEditable(True); self.c_fs_ema.setCurrentText("4096")
        gp.addWidget(self.c_fs_ema, 0, 1)

        gp.addWidget(QLabel("Número de Pontos (H1):"), 1, 0)
        self.c_pts_ema = QComboBox(); self.c_pts_ema.addItems(["512", "1024", "2048", "4096", "8192"]); self.c_pts_ema.setEditable(True); self.c_pts_ema.setCurrentText("2048")
        gp.addWidget(self.c_pts_ema, 1, 1)

        self.lbl_timing_ema = QLabel()
        gp.addWidget(self.lbl_timing_ema, 2, 0, 1, 2)
        
        self.c_fs_ema.currentTextChanged.connect(self.update_timing_ema)
        self.c_pts_ema.currentTextChanged.connect(self.update_timing_ema)
        self.update_timing_ema()
        
        gp.addWidget(QLabel("Mapeamento Modal 3D:"), 3, 0, 1, 2)
        self.c_map_base = QComboBox(); self.c_map_base.addItems(["Auto (Detectar pelo arquivo UFF)", "Nós de Resposta (Ex: Acelerômetros Móveis)", "Nós de Referência (Ex: Martelo Móvel)"])
        gp.addWidget(self.c_map_base, 3, 2)

        gp.addWidget(QLabel("Pré-Trigger (pontos):"), 4, 0)
        self.s_pretrig = QSpinBox(); self.s_pretrig.setRange(0, 500); self.s_pretrig.setValue(1)
        gp.addWidget(self.s_pretrig, 4, 1)

        gp.addWidget(QLabel("Freq. Mín/Máx (Hz):"), 5, 0)
        self.sp_fmin = QSpinBox(); self.sp_fmin.setRange(1, 10000); self.sp_fmin.setValue(10); gp.addWidget(self.sp_fmin, 5, 1)
        self.sp_fmax = QSpinBox(); self.sp_fmax.setRange(10, 20000); self.sp_fmax.setValue(600); gp.addWidget(self.sp_fmax, 5, 2)
        
        gp.addWidget(QLabel("Janelamento (Window):"), 6, 0)
        self.c_window = QComboBox(); self.c_window.addItems(["Retangular (Nenhum)", "Exponencial", "Hanning"])
        gp.addWidget(self.c_window, 6, 1, 1, 2)

        g_param.setLayout(gp)
        left_panel.addWidget(g_param)
        
        self.btn_run_ema = QPushButton("Extrair Modos (Abre pyEMA)")
        self.btn_run_ema.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_run_ema.clicked.connect(self.run_ema)
        left_panel.addWidget(self.btn_run_ema)
        
        g_visual = QGroupBox("Controles de Visualização 3D")
        gv = QGridLayout()
        gv.addWidget(QLabel("Direção da Deformação:"), 0, 0)
        self.c_eixo = QComboBox(); self.c_eixo.addItems(["Auto (Ler arquivo UFF)", "Forçar Bending em Z (Vertical)", "Forçar Bending em Y (Lateral)", "Forçar Bending em X (Axial)"]); self.c_eixo.setCurrentIndex(1)
        self.c_eixo.currentIndexChanged.connect(self.plot_3d_mode)
        gv.addWidget(self.c_eixo, 0, 1)

        gv.addWidget(QLabel("Multiplicador de Escala:"), 1, 0)
        self.s_escala = QDoubleSpinBox(); self.s_escala.setRange(0.1, 20.0); self.s_escala.setValue(1.0); self.s_escala.setSingleStep(0.5)
        self.s_escala.valueChanged.connect(self.plot_3d_mode)
        gv.addWidget(self.s_escala, 1, 1)
        g_visual.setLayout(gv)
        left_panel.addWidget(g_visual)
        
        self.c_view_mode = QComboBox()
        self.c_view_mode.addItem("Selecione o Modo após a Análise")
        self.c_view_mode.currentIndexChanged.connect(self.plot_3d_mode)
        left_panel.addWidget(QLabel("Visualizar Modo 3D:"))
        left_panel.addWidget(self.c_view_mode)
        left_panel.addStretch()
        
        right_panel = QVBoxLayout()
        self.fig_ema = Figure()
        self.canvas_ema = FigureCanvas(self.fig_ema)
        right_panel.addWidget(self.canvas_ema)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 3)

    def update_timing_ema(self):
        try:
            fs = float(self.c_fs_ema.currentText())
            pts = int(self.c_pts_ema.currentText())
            dur = pts / fs; df = fs / pts
            self.lbl_timing_ema.setText(f"Duração do Bloco: {dur:.3f} s | Resolução (Δf): {df:.3f} Hz")
            self.lbl_timing_ema.setStyleSheet("font-weight: bold; color: #0366d6;")
        except ValueError:
            self.lbl_timing_ema.setText("Valores Inválidos!")

    def load_uffs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Selecionar UFF", "", "UFF (*.uff)")
        for f in files:
            r = self.list_uffs.rowCount(); self.list_uffs.insertRow(r); self.list_uffs.setItem(r, 0, QTableWidgetItem(f))

    def clear_analysis_data(self):
        self.list_uffs.setRowCount(0)
        self.c_view_mode.clear()
        self.c_view_mode.addItem("Selecione o Modo após a Análise")
        if hasattr(self, 'A_shapes'): del self.A_shapes
        if hasattr(self, 'acc'): del self.acc
        self.fig_ema.clf()
        self.canvas_ema.draw()

    def run_ema(self):
        if not HAS_PYEMA: return QMessageBox.warning(self, "Aviso", "Módulo pyEMA não encontrado.")
            
        arquivos = [self.list_uffs.item(i, 0).text() for i in range(self.list_uffs.rowCount())]
        if not arquivos: return QMessageBox.warning(self, "Aviso", "Adicione arquivos .uff.")

        try:
            tamanho_bloco = int(self.c_pts_ema.currentText())
            fs_manual = float(self.c_fs_ema.currentText())
        except ValueError: return QMessageBox.warning(self, "Erro", "Valores inválidos.")

        pre_trigger = self.s_pretrig.value()
        tipo_janela = self.c_window.currentText()
        if "Exponencial" in tipo_janela: win = np.exp(-4.6 * np.linspace(0, 1, tamanho_bloco))
        elif "Hanning" in tipo_janela: win = np.hanning(tamanho_bloco)
        else: win = np.ones(tamanho_bloco)

        todas_frfs = []; self.frf_dofs = []; frequencias = np.fft.rfftfreq(tamanho_bloco, d=1/fs_manual)
        
        try:
            for arquivo in arquivos:
                datasets = pyuff.UFF(arquivo).read_sets()
                forcas = [ds for ds in datasets if ds.get('ordinate_spec_data_type') == 13]
                acels = [ds for ds in datasets if ds.get('ordinate_spec_data_type') == 12]
                if not forcas or not acels: continue
                
                ds_force = forcas[0]
                fs = 1.0 / ds_force['abscissa_inc']
                forca = ds_force['data']
                limite_forca = np.max(forca) * 0.15
                picos, _ = signal.find_peaks(forca, height=limite_forca, distance=tamanho_bloco//2)
                
                for ds_acc in acels:
                    aceleracao = ds_acc['data']
                    Sxx_soma = np.zeros(tamanho_bloco // 2 + 1, dtype=complex)
                    Sxy_soma = np.zeros(tamanho_bloco // 2 + 1, dtype=complex)
                    num_blocos = 0
                    
                    for pico in picos:
                        inicio = max(0, pico - pre_trigger)
                        fim = inicio + tamanho_bloco
                        if fim <= len(forca) and fim <= len(aceleracao):
                            f_seg = forca[inicio:fim]; a_seg = aceleracao[inicio:fim] * win
                            F = np.fft.rfft(f_seg); A = np.fft.rfft(a_seg)
                            Sxx_soma += F * np.conj(F); Sxy_soma += A * np.conj(F)
                            num_blocos += 1
                            
                    if num_blocos > 0:
                        Gxx = np.real(Sxx_soma / num_blocos); Gxx[Gxx == 0] = 1e-12 
                        todas_frfs.append((Sxy_soma / num_blocos) / Gxx)
                        self.frf_dofs.append({
                            'rsp_node': ds_acc['rsp_node'], 'rsp_dir': ds_acc['rsp_dir'], 
                            'ref_node': ds_acc['ref_node'], 'ref_dir': ds_acc['ref_dir']
                        })
                        
            if not todas_frfs: return QMessageBox.warning(self, "Aviso", "Nenhuma FRF calculada.")

            tamanho_minimo = min(len(frf) for frf in todas_frfs)
            todas_frfs = [frf[:tamanho_minimo] for frf in todas_frfs]; frequencias = frequencias[:tamanho_minimo]

            self.acc = pyEMA.Model(frf=np.array(todas_frfs), freq=frequencias, lower=self.sp_fmin.value(), upper=self.sp_fmax.value(), pol_order_high=60)
            self.acc.get_poles()
            
            plt.show = matplotlib.pyplot.show
            QMessageBox.information(self, "Seleção", "Selecione os polos (X verdes) e feche a janela do pyEMA.")
            self.acc.select_poles()
            plt.show = lambda *args, **kwargs: None
            
            if not hasattr(self.acc, 'nat_freq') or len(self.acc.nat_freq) == 0: return
            self.H_synth, self.A_shapes = self.acc.get_constants(whose_poles='own')
            
            self.c_view_mode.blockSignals(True)
            self.c_view_mode.clear()
            for i, (fn, zeta) in enumerate(zip(self.acc.nat_freq, self.acc.nat_xi)):
                self.c_view_mode.addItem(f"Modo {i+1}: {fn:.2f} Hz, amort: {zeta*100:.2f}%")
            self.c_view_mode.blockSignals(False)
            
            self.plot_3d_mode()

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro nas FRFs:\n{str(e)}")

    def plot_3d_mode(self):
        mode_idx = self.c_view_mode.currentIndex()
        if mode_idx < 0 or not hasattr(self, 'A_shapes'): return
        
        nodes, lines = self.get_geometry_data()
        self.fig_ema.clf()
        ax = self.fig_ema.add_subplot(111, projection='3d')
        
        for n1, n2 in lines:
            if n1 in nodes and n2 in nodes:
                p1, p2 = nodes[n1], nodes[n2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', linestyle='--', alpha=0.3, lw=1.5)
                
        node_disp = {nid: np.zeros(3) for nid in nodes}
        map_text = self.c_map_base.currentText()
        usar_ref = False
        
        if "Auto" in map_text:
            refs = [d['ref_node'] for d in self.frf_dofs]
            rsps = [d['rsp_node'] for d in self.frf_dofs]
            if len(set(refs)) > 1 and len(set(rsps)) == 1: usar_ref = True
            else: usar_ref = False
        elif "Referência" in map_text: usar_ref = True
            
        forcar_eixo = self.c_eixo.currentIndex() 
        
        for i, dof in enumerate(self.frf_dofs):
            if i >= self.A_shapes.shape[0]: break
            nid = dof['ref_node'] if usar_ref else dof['rsp_node']
            direction = dof['ref_dir'] if usar_ref else dof['rsp_dir']
            
            amp = self.A_shapes[i, mode_idx]
            ref_idx = np.argmax(np.abs(self.A_shapes[:, mode_idx]))
            real_amp = np.real(amp * np.exp(-1j * np.angle(self.A_shapes[ref_idx, mode_idx])))
            
            if nid in node_disp:
                if forcar_eixo == 1: node_disp[nid][2] += real_amp
                elif forcar_eixo == 2: node_disp[nid][1] += real_amp
                elif forcar_eixo == 3: node_disp[nid][0] += real_amp
                else: 
                    if direction in [1, -1]: node_disp[nid][0] += real_amp * np.sign(direction)
                    elif direction in [2, -2]: node_disp[nid][1] += real_amp * np.sign(direction)
                    elif direction in [3, -3]: node_disp[nid][2] += real_amp * np.sign(direction)

        if nodes:
            coords = np.array(list(nodes.values()))
            max_geom = np.max(np.ptp(coords, axis=0)) if np.max(np.ptp(coords, axis=0)) > 0 else 100
        else: max_geom = 100
            
        max_d = max([np.linalg.norm(d) for d in node_disp.values()]) if node_disp else 0
        scale = (0.20 * max_geom / max_d) * self.s_escala.value() if max_d > 0 else 1.0 
        
        for nid in nodes:
            if np.linalg.norm(node_disp[nid]) > 0:
                p_undef = np.array(nodes[nid])
                p_def = p_undef + scale * node_disp[nid]
                ax.plot([p_undef[0], p_def[0]], [p_undef[1], p_def[1]], [p_undef[2], p_def[2]], color='orange', linestyle=':', lw=1.5)
                ax.plot([p_undef[0], p_def[0]], [p_undef[1], p_def[1]], [p_undef[2], p_def[2]], color='green', linestyle=':', lw=1.0, alpha=0.5)

        for n1, n2 in lines:
            if n1 in nodes and n2 in nodes:
                p1 = np.array(nodes[n1]) + scale * node_disp[n1]
                p2 = np.array(nodes[n2]) + scale * node_disp[n2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='blue', linewidth=3)
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red', s=50, edgecolors='black', zorder=5)
                
        ax.set_title(self.c_view_mode.currentText(), pad=15, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (mm)', fontweight='bold'); ax.set_ylabel('Y (mm)', fontweight='bold'); ax.set_zlabel('Z (mm)', fontweight='bold')
        
        if nodes:
            all_x, all_y, all_z = [], [], []
            for nid in nodes:
                undef = np.array(nodes[nid]); p_def = undef + scale * node_disp[nid]
                all_x.extend([undef[0], p_def[0]]); all_y.extend([undef[1], p_def[1]]); all_z.extend([undef[2], p_def[2]])
            dx = max(max(all_x) - min(all_x), 1.0); dy = max(max(all_y) - min(all_y), 1.0); dz = max(max(all_z) - min(all_z), 1.0)
            ax.set_xlim(min(all_x) - dx*0.1, max(all_x) + dx*0.1); ax.set_ylim(min(all_y) - dy*0.1, max(all_y) + dy*0.1); ax.set_zlim(min(all_z) - dz*0.1, max(all_z) + dz*0.1)
            try: ax.set_box_aspect((dx, max(dy, max(dx,dy,dz)*0.3), max(dz, max(dx,dy,dz)*0.3)))
            except AttributeError: pass
            
        ax.view_init(elev=20, azim=-60)
        self.fig_ema.tight_layout()
        self.canvas_ema.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    visualizer = EMAVisualizer()
    visualizer.show()
    sys.exit(app.exec_())