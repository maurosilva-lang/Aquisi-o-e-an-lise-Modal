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
# IMPORTS E TRATAMENTO DE EXCEÇÕES
# ===================================================================
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                                 QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                                 QLineEdit, QComboBox, QCheckBox, QPushButton, 
                                 QTableWidget, QTableWidgetItem, QHeaderView, 
                                 QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox,
                                 QFileDialog, QFrame)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    
    import matplotlib
    matplotlib.use('Qt5Agg') 
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

# AJUSTE 1: Capturando qualquer tipo de exceção e guardando a mensagem real
try:
    import nidaqmx
    from nidaqmx.constants import (AcquisitionType, AccelSensitivityUnits, 
                                  ForceIEPESensorSensitivityUnits, ExcitationSource)
    HAS_NIDAQ = True
    NIDAQ_ERR_MSG = ""
except Exception as e:
    HAS_NIDAQ = False
    NIDAQ_ERR_MSG = str(e)

def get_disp_unit(unit_str):
    if unit_str == 'mV/g': return 'g'
    if unit_str == 'mV/(m/s²)': return 'm/s²'
    if unit_str == 'mV/N': return 'N'
    if unit_str == 'mV/lbf': return 'lbf'
    return ''

# ===================================================================
# THREAD DE AQUISIÇÃO (APENAS GRAVAÇÃO FINITA)
# ===================================================================
class AcquisitionThread(QThread):
    data_signal = pyqtSignal(np.ndarray, np.ndarray, bool) 
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

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
            self.error_signal.emit(f"Erro de hardware/driver: {NIDAQ_ERR_MSG}")
            self.finished_signal.emit()
            return
            
        fs = self.config['fs']
        nsamples = self.config['nsamples']
        duration = nsamples / fs
        channels = self.config['channels']
        
        try:
            with nidaqmx.Task() as task:
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
                        if ch.get('unit') == 'mV/lbf': sens_val = sens_val / 4.4482216
                        task.ai_channels.add_ai_force_iepe_chan(
                            physical_channel=physical_ch, min_val=-200.0, max_val=200.0,
                            sensitivity=sens_val, sensitivity_units=ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,  
                            current_excit_source=excit_src, current_excit_val=excit_val   
                        )
                
                task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=nsamples)
                task.start()
                data_chunk = task.read(number_of_samples_per_channel=nsamples, timeout=duration + 10.0)
                data_array = np.array(data_chunk)
                if len(channels) == 1: data_array = data_array.reshape((1, -1))
                task.stop()
                
                if not self.is_running:
                    self.finished_signal.emit()
                    return
                    
                data_array = self.remove_dc_offset(data_array)
                data_array = self.apply_units(data_array, channels)
                t = np.linspace(0, duration, nsamples, endpoint=False)
                
                self.data_signal.emit(t, data_array, False)
                self.finished_signal.emit()
                return

        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished_signal.emit()

# ===================================================================
# INTERFACE GRÁFICA PRINCIPAL
# ===================================================================
class DataAcqVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plataforma de Aquisição de Sinais e FFT")
        self.resize(1350, 950)
        
        # O hardware agora é detectado automaticamente e permite hot-plugging
        self.hardware_channels = self.auto_detect_channels()
                
        self.acq_thread = None
        self.last_saved_txt = None  
        self.last_fft_data = None
        
        self.build_main_app()

    def auto_detect_channels(self):
        detected_channels = []
        if HAS_NIDAQ:
            try:
                system = nidaqmx.system.System.local()
                ch_index = 0
                
                # AJUSTE: Usar system.devices.device_names forca a C-API a varrer o hardware atual
                for dev_name in system.devices.device_names:
                    dev = system.devices[dev_name]
                    if not dev.is_simulated:
                        try:
                            # Varre todos os canais de input analógico da placa encontrada
                            for ai_chan in dev.ai_physical_chans:
                                ch_name = ai_chan.name
                                if '/' in ch_name:
                                    dev_name_ch, chan_name = ch_name.split('/', 1)
                                else:
                                    dev_name_ch, chan_name = dev.name, ch_name
                                
                                detected_channels.append({
                                    'device': dev_name_ch, 
                                    'channel': chan_name,
                                    'enabled': True if ch_index < 2 else False, # Deixa os 2 primeiros ativados por padrão
                                    'type': 'Aceleração', 
                                    'label': f"Canal {ch_index + 1}", 
                                    'sensitivity': 100.0,
                                    'unit': 'mV/g', 
                                    'iepe': 'Ativado (2mA)',
                                    'node': ch_index + 1, 
                                    'dir': '+Z'
                                })
                                ch_index += 1
                        except Exception:
                            continue # Pula o dispositivo se ele não suportar canais analógicos
            except Exception as e:
                print(f"Erro ao detectar canais: {e}")
        
        # Fallback genérico caso nenhuma placa seja encontrada
        if not detected_channels:
            detected_channels = [
                {'device': 'cDAQ1Mod1', 'channel': 'ai0', 'enabled': True, 'type': 'Aceleração', 'label': 'Canal 1', 'sensitivity': 100.0, 'unit': 'mV/g', 'iepe': 'Ativado (2mA)', 'node': 1, 'dir': '+Z'},
                {'device': 'cDAQ1Mod1', 'channel': 'ai1', 'enabled': True, 'type': 'Aceleração', 'label': 'Canal 2', 'sensitivity': 100.0, 'unit': 'mV/g', 'iepe': 'Ativado (2mA)', 'node': 2, 'dir': '+Z'}
            ]
        return detected_channels

    def build_main_app(self):
        self.main_app_widget = QWidget()
        self.setCentralWidget(self.main_app_widget)
        main_layout = QVBoxLayout(self.main_app_widget)
        
        global_frame = QFrame()
        global_frame.setMaximumHeight(60) 
        global_layout = QHBoxLayout(global_frame)
        global_layout.setContentsMargins(5, 5, 5, 5)
        
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
        self.tab_ch = QWidget(); self.setupTab_Channels()
        self.tab_acq = QWidget(); self.setupTab_Acquisition()
        self.tab_post = QWidget(); self.setupTab_PostProcessing()
        self.tabs.addTab(self.tab_ch, "1. Mapeamento de Canais")
        self.tabs.addTab(self.tab_acq, "2. Aquisição de Dados")
        self.tabs.addTab(self.tab_post, "3. Pós-Processamento (FFT)")
        main_layout.addWidget(self.tabs)
        self.check_hardware_connection(manual=False)

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
        err_detail = ""
        
        if HAS_NIDAQ:
            try:
                system = nidaqmx.system.System.local()
                active_devs = []
                # AJUSTE: Usar system.devices.device_names forca a C-API a varrer o hardware atualizado
                for dev_name in system.devices.device_names:
                    dev = system.devices[dev_name]
                    if not dev.is_simulated:
                        active_devs.append(f"{dev.name} ({dev.product_type})")
                
                if active_devs:
                    msg = f"✅ Hardware OK: {', '.join(active_devs)}"
                    connected = True
                else:
                    msg = "⚠️ Driver OK, mas nenhum dispositivo físico foi listado pelo sistema."
            except Exception as e:
                err_detail = str(e)
                msg = f"❌ Falha no driver DAQmx:\n{err_detail}"
        else:
            msg = f"❌ Erro nidaqmx no .exe: {NIDAQ_ERR_MSG}"
            
        self.lbl_status.setText(msg.split('\n')[0])
        
        if manual:
            # Se for manual (pressionou 'Atualizar Hardware'), reescaneia e atualiza a tabela
            self.hardware_channels = self.auto_detect_channels()
            self.table_instruments.setRowCount(0)
            for ch in self.hardware_channels:
                self.add_instrument_row(ch)

        if connected:
            self.lbl_status.setStyleSheet("background-color: #2da44e; color: white; padding: 5px; font-weight: bold;")
            if manual: QMessageBox.information(self, "Status", msg)
        else:
            self.lbl_status.setStyleSheet("background-color: #cf222e; color: white; padding: 5px; font-weight: bold;")
            if manual: QMessageBox.warning(self, "Status", msg)

    # ===============================================================
    # ABA 1: MAPEAMENTO DE CANAIS
    # ===============================================================
    def setupTab_Channels(self):
        layout = QVBoxLayout(self.tab_ch)
        self.table_instruments = QTableWidget(0, 8)
        self.table_instruments.setHorizontalHeaderLabels(["Ativar", "Dispositivo/Canal", "Tipo", "Ponto", "Direção", "Sensibilidade", "Unidade", "IEPE"])
        self.table_instruments.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_instruments)
        
        ch_btn_layout = QHBoxLayout()
        btn_add_ch = QPushButton("+ Adicionar Canal"); btn_add_ch.clicked.connect(lambda: self.add_instrument_row())
        btn_del_ch = QPushButton("- Remover Canal"); btn_del_ch.clicked.connect(self.delete_instrument_row)
        ch_btn_layout.addWidget(btn_add_ch); ch_btn_layout.addWidget(btn_del_ch)
        layout.addLayout(ch_btn_layout)
        
        # Agora injetamos TODOS os canais que a detecção automática encontrou
        for ch in self.hardware_channels:
            self.add_instrument_row(ch)
            
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Carregar (.json)"); btn_load.clicked.connect(self.load_config)
        btn_save = QPushButton("Salvar (.json)"); btn_save.clicked.connect(self.save_config)
        btn_layout.addWidget(btn_load); btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

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

    def save_config(self):
        config = []
        for i in range(self.table_instruments.rowCount()):
            parts = self.table_instruments.cellWidget(i, 1).text().split('/')
            config.append({
                'enabled': self.table_instruments.cellWidget(i, 0).isChecked(), 'device': parts[0], 'channel': parts[1] if len(parts) > 1 else 'ai0',
                'type': self.table_instruments.cellWidget(i, 2).currentText(), 'node': self.table_instruments.cellWidget(i, 3).value(),
                'dir': self.table_instruments.cellWidget(i, 4).currentText(), 'sensitivity': self.table_instruments.cellWidget(i, 5).value(),
                'unit': self.table_instruments.cellWidget(i, 6).currentText(),
                'iepe': self.table_instruments.cellWidget(i, 7).currentText()
            })
        filename, _ = QFileDialog.getSaveFileName(self, "Salvar Configuração", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w', encoding='utf-8') as f: json.dump(config, f, ensure_ascii=False, indent=4)

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Carregar Configurações", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'r', encoding='utf-8') as f: loaded = json.load(f)
            self.table_instruments.setRowCount(0)
            for ch in loaded: self.add_instrument_row(ch)

    def get_active_channels(self):
        active = []
        for i in range(self.table_instruments.rowCount()):
            chk = self.table_instruments.cellWidget(i, 0)
            combo = self.table_instruments.cellWidget(i, 2)
            if chk and chk.isChecked() and combo.currentText() != "Desativado":
                parts = self.table_instruments.cellWidget(i, 1).text().split('/')
                active.append({
                    'device': parts[0], 'channel': parts[1] if len(parts) > 1 else 'ai0', 
                    'type': combo.currentText(), 'label': f"Sinal {i+1}", 
                    'sensitivity': self.table_instruments.cellWidget(i, 5).value(),
                    'unit': self.table_instruments.cellWidget(i, 6).currentText(),
                    'iepe': self.table_instruments.cellWidget(i, 7).currentText()
                })
        return active

    # ===============================================================
    # ABA 2: AQUISIÇÃO
    # ===============================================================
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
        self.c_pts_acq.addItems(["512", "1024", "2048", "4096", "8192", "16384", "32768"])
        self.c_pts_acq.setEditable(True); self.c_pts_acq.setCurrentText("8192")
        grid_timing.addWidget(self.c_pts_acq, 1, 1)

        self.lbl_timing_acq = QLabel()
        grid_timing.addWidget(self.lbl_timing_acq, 2, 0, 1, 2)
        group_timing.setLayout(grid_timing)
        left_panel.addWidget(group_timing)

        self.c_fs_acq.currentTextChanged.connect(self.update_timing_acq)
        self.c_pts_acq.currentTextChanged.connect(self.update_timing_acq)
        self.update_timing_acq()
        
        self.btn_record_acq = QPushButton("Extrair e Gravar Sinal (.txt)")
        self.btn_record_acq.setStyleSheet("background-color: #2da44e; color: white; min-height: 50px; font-size: 14px; font-weight: bold;")
        self.btn_record_acq.clicked.connect(self.start_record)
        left_panel.addWidget(self.btn_record_acq)
        
        self.btn_stop = QPushButton("Cancelar Aquisição")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_acquisition)
        left_panel.addWidget(self.btn_stop)
        left_panel.addStretch()
        
        right_panel = QVBoxLayout()
        self.figure = Figure()
        self.canvas_sig = FigureCanvas(self.figure)
        
        self.toolbar_acq = NavigationToolbar(self.canvas_sig, self)
        right_panel.addWidget(self.toolbar_acq)
        right_panel.addWidget(self.canvas_sig)
        
        self.ax_acq = self.figure.add_subplot(111)
        self.ax_acq.set_title("Resposta no Tempo")
        self.ax_acq.set_ylabel("Amplitude")
        self.ax_acq.set_xlabel("Tempo [s]")
        self.ax_acq.grid(True)
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

    def start_record(self):
        active_channels = self.get_active_channels()
        if not active_channels: return QMessageBox.warning(self, "Erro", "Nenhum canal ativo!")
        config = {
            'fs': float(self.c_fs_acq.currentText()), 'nsamples': int(self.c_pts_acq.currentText()),
            'channels': active_channels, 'acq_type': 'record'
        }
        self.btn_record_acq.setText("Adquirindo Dados...")
        self.btn_record_acq.setStyleSheet("background-color: #d4a72c; color: black; min-height: 50px; font-size: 14px; font-weight: bold;")
        self.btn_record_acq.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
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

    def handle_error(self, err_msg):
        QMessageBox.critical(self, "Erro na Placa", f"Erro:\n{err_msg}")
        self.acquisition_finished()

    def acquisition_finished(self):
        self.btn_record_acq.setText("Extrair e Gravar Sinal (.txt)")
        self.btn_record_acq.setStyleSheet("background-color: #2da44e; color: white; min-height: 50px; font-size: 14px; font-weight: bold;")
        self.btn_record_acq.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def route_data(self, time_axis, data_matrix, double_hit):
        self.ax_acq.clear()
        active = self.get_active_channels()
        fs = float(self.c_fs_acq.currentText())
        labels_plot = [f"{ch['label']} [{get_disp_unit(ch.get('unit', ''))}]" for ch in active]
        
        for idx in range(data_matrix.shape[0]):
            self.ax_acq.plot(time_axis, data_matrix[idx, :], alpha=0.8, label=labels_plot[idx])
            
        self.ax_acq.set_title("Resposta no Tempo")
        self.ax_acq.set_ylabel("Amplitude")
        self.ax_acq.set_xlabel("Tempo [s]")
        self.ax_acq.grid(True)
        self.ax_acq.legend(loc="upper right")
        self.canvas_sig.draw()
        
        self.process_and_plot_post(time_axis, data_matrix, labels_plot, fs)
        self.tabs.setCurrentIndex(2) 
        
        reply = QMessageBox.question(self, "Salvar", "Aquisição finalizada.\nDeseja salvar a série temporal em um arquivo .txt agora?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            filename, _ = QFileDialog.getSaveFileName(self, "Salvar Sinal Temporal", f"Dados_{datetime.datetime.now().strftime('%H%M%S')}.txt", "Text Files (*.txt)")
            if filename:
                try:
                    header = "Time(s)\t" + "\t".join(labels_plot)
                    out_data = np.vstack((time_axis, data_matrix)).T
                    # Adicionado encoding='latin1' aqui também para garantir o formato correto ao salvar futuramente
                    np.savetxt(filename, out_data, delimiter='\t', header=header, comments='', encoding='latin1')
                    self.lbl_loaded_file.setText(f"Arquivo: {os.path.basename(filename)}")
                    QMessageBox.information(self, "Sucesso", "Dados exportados com sucesso!")
                except Exception as e: QMessageBox.critical(self, "Erro", str(e))

    # ===============================================================
    # ABA 3: PÓS-PROCESSAMENTO (FFT)
    # ===============================================================
    def setupTab_PostProcessing(self):
        layout = QHBoxLayout(self.tab_post)
        left_panel = QVBoxLayout()
        
        group_file = QGroupBox("Gerenciamento de Arquivos")
        file_layout = QVBoxLayout()
        self.btn_load_txt = QPushButton("📂 Abrir Arquivo (.txt)")
        self.btn_load_txt.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold; min-height: 35px;")
        self.btn_load_txt.clicked.connect(self.load_external_txt)
        file_layout.addWidget(self.btn_load_txt)
        
        self.lbl_loaded_file = QLabel("Nenhum arquivo carregado.")
        self.lbl_loaded_file.setAlignment(Qt.AlignCenter)
        file_layout.addWidget(self.lbl_loaded_file)
        group_file.setLayout(file_layout)
        left_panel.addWidget(group_file)
        
        group_stats = QGroupBox("Estatísticas do Sinal")
        stats_layout = QVBoxLayout()
        self.table_stats = QTableWidget(0, 4)
        self.table_stats.setHorizontalHeaderLabels(["Canal", "Pico", "RMS", "F. Pico [Hz]"])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.table_stats)
        group_stats.setLayout(stats_layout)
        left_panel.addWidget(group_stats)
        left_panel.addStretch()
        
        right_panel = QVBoxLayout()
        self.fig_post = Figure()
        self.ax_post_time = self.fig_post.add_subplot(211)
        self.ax_post_fft = self.fig_post.add_subplot(212)
        self.canvas_post = FigureCanvas(self.fig_post)
        
        top_right_layout = QHBoxLayout()
        self.toolbar_post = NavigationToolbar(self.canvas_post, self)
        top_right_layout.addWidget(self.toolbar_post)
        
        top_right_layout.addStretch()
        
        self.chk_db = QCheckBox("Escala em dB (FFT)")
        self.chk_db.stateChanged.connect(self.replot_fft)
        top_right_layout.addWidget(self.chk_db)
        
        top_right_layout.addWidget(QLabel("Ref:"))
        self.txt_db_ref = QLineEdit("1.0")
        self.txt_db_ref.setFixedWidth(60)
        self.txt_db_ref.textChanged.connect(self.replot_fft)
        top_right_layout.addWidget(self.txt_db_ref)
        
        right_panel.addLayout(top_right_layout)
        right_panel.addWidget(self.canvas_post)
        
        self.ax_post_time.set_title("Sinal no Tempo")
        self.ax_post_time.grid(True)
        self.ax_post_fft.set_title("Espectro de Frequência (FFT)")
        self.ax_post_fft.grid(True)
        self.fig_post.tight_layout()
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 3)

    def load_external_txt(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Arquivo Temporal", "", "Text Files (*.txt)")
        if filename:
            try:
                # AJUSTE 2: O np.loadtxt() tamém precisa receber o encoding latin1
                with open(filename, 'r', encoding='latin1') as f:
                    header = f.readline().strip()
                labels = header.split('\t')[1:]
                if not labels: labels = ["Canal 1"]
                
                # Numpy abrirá o arquivo de forma segura usando o latin1
                data = np.loadtxt(filename, delimiter='\t', skiprows=1, encoding='latin1')
                
                time_axis = data[:, 0]; data_matrix = data[:, 1:].T
                if len(data_matrix.shape) == 1: data_matrix = data_matrix.reshape(1, -1)
                fs = 1.0 / (time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 1000.0
                self.process_and_plot_post(time_axis, data_matrix, labels, fs)
                self.lbl_loaded_file.setText(f"Carregado: {os.path.basename(filename)}")
            except Exception as e: QMessageBox.critical(self, "Erro", str(e))

    def process_and_plot_post(self, time_axis, data_matrix, labels, fs):
        self.ax_post_time.clear()
        self.table_stats.setRowCount(0)
        
        self.last_fft_data = {'xf': None, 'amps': [], 'labels': labels}
        
        for idx in range(data_matrix.shape[0]):
            sig = data_matrix[idx, :]
            lbl = labels[idx] if idx < len(labels) else f"Canal {idx+1}"
            
            self.ax_post_time.plot(time_axis, sig, alpha=0.8, label=lbl)
            
            N = len(sig)
            yf = np.fft.rfft(sig)
            xf = np.fft.rfftfreq(N, d=1/fs)
            amp = 2.0/N * np.abs(yf)
            
            self.last_fft_data['xf'] = xf
            self.last_fft_data['amps'].append(amp)
            
            rms = np.sqrt(np.mean(sig**2))
            peak = np.max(np.abs(sig))
            peak_freq = xf[np.argmax(amp)]
            
            row = self.table_stats.rowCount()
            self.table_stats.insertRow(row)
            self.table_stats.setItem(row, 0, QTableWidgetItem(lbl))
            self.table_stats.setItem(row, 1, QTableWidgetItem(f"{peak:.4f}"))
            self.table_stats.setItem(row, 2, QTableWidgetItem(f"{rms:.4f}"))
            self.table_stats.setItem(row, 3, QTableWidgetItem(f"{peak_freq:.1f}"))
            
        self.ax_post_time.set_title("Sinal no Tempo")
        self.ax_post_time.set_ylabel("Amplitude")
        self.ax_post_time.set_xlabel("Tempo [s]")
        self.ax_post_time.grid(True)
        self.ax_post_time.legend(loc="upper right")
        
        self.replot_fft()

    def replot_fft(self):
        if not hasattr(self, 'last_fft_data') or not self.last_fft_data['amps']: return
            
        self.ax_post_fft.clear()
        use_db = self.chk_db.isChecked()
        
        try:
            ref_val = float(self.txt_db_ref.text().replace(',', '.'))
            if ref_val <= 0: ref_val = 1.0
        except ValueError:
            ref_val = 1.0
            
        xf = self.last_fft_data['xf']
        for i, amp in enumerate(self.last_fft_data['amps']):
            lbl = self.last_fft_data['labels'][i]
            if use_db:
                amp_plot = 20 * np.log10((amp + 1e-12) / ref_val)
            else:
                amp_plot = amp
            self.ax_post_fft.plot(xf, amp_plot, alpha=0.8, label=lbl)
            
        self.ax_post_fft.set_title("Espectro de Frequência (FFT)")
        self.ax_post_fft.set_ylabel("Amplitude [dB]" if use_db else "Amplitude Linear")
        self.ax_post_fft.set_xlabel("Frequência [Hz]")
        self.ax_post_fft.grid(True)
        self.ax_post_fft.legend(loc="upper right")
        
        self.fig_post.tight_layout()
        self.canvas_post.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    visualizer = DataAcqVisualizer()
    visualizer.show()
    sys.exit(app.exec_())