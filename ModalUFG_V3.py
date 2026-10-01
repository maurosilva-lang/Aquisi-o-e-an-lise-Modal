import sys
import numpy as np
import datetime
import json
import os
import scipy.signal as signal
import re
import time
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
                                 QFileDialog, QFrame, QStackedWidget)
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

# Importando Modalyzer para OMA
try:
    import pandas as pd
    import torch
    from modalyzer import preprocess
    from modalyzer.EFDD import FDD_SVD_Diagram, FDD_ModeShape, FDD_Damping
    from modalyzer.SSI import SSI_Alg, psd_func, svd_psd
    HAS_MODALYZER = True
except ImportError:
    HAS_MODALYZER = False

# ===================================================================
# FUNÇÃO AUXILIAR DE UNIDADES DE EXIBIÇÃO
# ===================================================================
def get_disp_unit(unit_str):
    if unit_str == 'mV/g': return 'g'
    if unit_str == 'mV/(m/s²)': return 'm/s²'
    if unit_str == 'mV/N': return 'N'
    if unit_str == 'mV/lbf': return 'lbf'
    return ''

# ===================================================================
# THREAD DE AQUISIÇÃO - ROBUSTA COM MODOS INDEPENDENTES
# ===================================================================
class AcquisitionThread(QThread):
    data_signal = pyqtSignal(np.ndarray, np.ndarray, bool) 
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    oma_progress_signal = pyqtSignal(float)
    oma_finished_signal = pyqtSignal(np.ndarray)

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
        # Converte os dados lidos (g ou N) para a unidade de exibição/salvamento solicitada
        for idx, ch in enumerate(channels):
            if ch['type'] == 'Aceleração' and ch.get('unit') == 'mV/(m/s²)':
                data[idx, :] *= 9.80665
            elif ch['type'] == 'Força' and ch.get('unit') == 'mV/lbf':
                data[idx, :] /= 4.4482216
        return data

    def run(self):
        fs = self.config['fs']
        channels = self.config['channels']
        acq_type = self.config.get('acq_type', 'ema') 
        
        # =========================================================
        # FLUXO OMA: SIMULAÇÃO OU HARDWARE
        # =========================================================
        if 'oma' in acq_type:
            is_preview = (acq_type == 'oma_monitor')
            is_sim = self.config.get('simulation', False)
            duration_min = self.config.get('duration_min', 1.0)
            total_time = duration_min * 60.0
            total_samples = int(fs * total_time)
            chunk_size = int(fs * 0.25)
            
            full_data = []
            samples_read = 0
            
            # --- OMA: MODO SIMULAÇÃO ---
            if not HAS_NIDAQ or is_sim:
                freqs = [24.5, 75.2, 142.1]
                dampings = [0.015, 0.02, 0.025]
                
                while self.is_running:
                    if not is_preview and samples_read >= total_samples: break
                    
                    t_chunk = np.linspace(samples_read/fs, (samples_read+chunk_size)/fs, chunk_size, endpoint=False)
                    data_chunk = np.zeros((len(channels), chunk_size))
                    ambient_noise = np.random.normal(0, 2.0, chunk_size)

                    for idx, ch in enumerate(channels):
                        if ch['type'] == 'Aceleração':
                            response = ambient_noise * (0.5 / (idx + 1))
                            for f, zeta in zip(freqs, dampings):
                                response += np.sin(2 * np.pi * f * t_chunk) * np.random.normal(0, 0.08)
                            data_chunk[idx, :] = response

                    data_chunk = self.remove_dc_offset(data_chunk)
                    data_chunk = self.apply_units(data_chunk, channels)
                    
                    if not is_preview: full_data.append(data_chunk)
                    
                    samples_read += chunk_size
                    self.data_signal.emit(t_chunk, data_chunk, False)
                    
                    if not is_preview: self.oma_progress_signal.emit(samples_read / fs)
                    time.sleep(0.5)

                if not is_preview and full_data: self.oma_finished_signal.emit(np.concatenate(full_data, axis=1))
                elif is_preview: self.oma_finished_signal.emit(np.array([]))
                return

            # --- OMA: MODO HARDWARE FÍSICO ---
            else:
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
                                    sensitivity=sens_val, 
                                    sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,  
                                    current_excit_source=excit_src, current_excit_val=excit_val   
                                )
                                
                        task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
                        task.start()
                        
                        while self.is_running:
                            if not is_preview and samples_read >= total_samples: break
                            
                            raw_data = task.read(number_of_samples_per_channel=chunk_size, timeout=10.0)
                            data_array = np.array(raw_data)
                            if len(channels) == 1: data_array = data_array.reshape((1, -1))
                            data_array = self.remove_dc_offset(data_array)
                            data_array = self.apply_units(data_array, channels)
                            
                            if not is_preview: full_data.append(data_array)
                            
                            t_chunk = np.linspace(samples_read/fs, (samples_read+chunk_size)/fs, chunk_size, endpoint=False)
                            samples_read += chunk_size
                            
                            self.data_signal.emit(t_chunk, data_array, False)
                            if not is_preview: self.oma_progress_signal.emit(samples_read / fs)
                            
                        task.stop()
                        if not is_preview and full_data: self.oma_finished_signal.emit(np.concatenate(full_data, axis=1))
                        elif is_preview: self.oma_finished_signal.emit(np.array([]))
                        return
                except Exception as e:
                    self.error_signal.emit(str(e))
                    if not is_preview and full_data: self.oma_finished_signal.emit(np.concatenate(full_data, axis=1))
                    return

        # =========================================================
        # FLUXOS ACQ (Aquisição Simples) e EMA (Com Martelo)
        # =========================================================
        if not HAS_NIDAQ:
            self.error_signal.emit("A biblioteca 'nidaqmx' não está instalada ou o hardware não foi detectado.")
            self.finished_signal.emit()
            return
            
        nsamples = self.config['nsamples']
        duration = nsamples / fs
        
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
                
                # --- ACQ MONITORAMENTO CONTÍNUO ---
                if acq_type == 'monitor':
                    task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
                    task.start()
                    chunk_size = int(fs * 0.25) 
                    while self.is_running:
                        data_chunk = task.read(number_of_samples_per_channel=chunk_size, timeout=2.0)
                        data_array = np.array(data_chunk)
                        if len(channels) == 1: data_array = data_array.reshape((1, -1))
                        data_array = self.remove_dc_offset(data_array)
                        data_array = self.apply_units(data_array, channels)
                        self.data_signal.emit(np.array([]), data_array, False)
                    task.stop()
                    self.finished_signal.emit()
                    return

                # --- ACQ GRAVAÇÃO FINITA ---
                elif acq_type == 'record':
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

                # --- EMA SOFTWARE TRIGGER ---
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
                    
                    # Aplica a conversão apenas para detecção visual (sem perder o bruto), o buffer recebe os dados puros para o trigger e depois aplicamos unidades globais
                    
                    if not self.config['use_trigger'] or hammer_idx_real == -1:
                        buffer = np.hstack((buffer, chunk_np))
                        if buffer.shape[1] >= total_samps:
                            data_array = buffer[:, :total_samps]
                            break
                        continue
                        
                    hammer_chunk = chunk_np[hammer_idx_real, :]
                    hammer_chunk_ac = hammer_chunk - np.mean(hammer_chunk)
                    
                    # Converte o limiar se a unidade visual for lbf
                    trigger_val_real = self.config['trigger_level']
                    
                    if not triggered:
                        if np.max(np.abs(hammer_chunk_ac)) >= trigger_val_real:
                            triggered = True
                            buffer = np.hstack((buffer, chunk_np))
                            
                            samps_to_read = total_samps
                            rest_data = task.read(number_of_samples_per_channel=samps_to_read, timeout=duration + 5.0)
                            rest_np = np.array(rest_data)
                            if len(channels) == 1: rest_np = rest_np.reshape((1, -1))
                            
                            buffer = np.hstack((buffer, rest_np))
                            
                            full_hammer = buffer[hammer_idx_real, :]
                            full_hammer_ac = full_hammer - np.mean(full_hammer)
                            trig_indices = np.where(np.abs(full_hammer_ac) >= trigger_val_real)[0]
                            
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

# ===================================================================
# INTERFACE GRÁFICA PRINCIPAL
# ===================================================================
class MAVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plataforma EMA/OMA 3D - Análise Estrutural")
        self.resize(1400, 950)
        
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
        self.monitor_buffer = None
        self.acq_thread = None
        self.current_mode = 'ema'
        self.current_acq_type = 'ema'
        self.last_saved_txt = None  
        
        # Variáveis OMA (Pós-processamento)
        self.oma_data_raw = None
        self.oma_time_raw = None
        self.oma_labels = []
        self.oma_fs = 1000.0
        self.oma_dt_novo = 0.001
        self.oma_dados_efdd = None
        self.oma_psd_mat = None
        self.oma_f_psd = None
        self.oma_ss1 = None
        self.oma_UU = None
        self.oma_t_out = None
        self.oma_dofs_parsed = []
        
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.build_start_screen()
        self.build_main_app()
        
        self.stacked_widget.addWidget(self.start_screen)
        self.stacked_widget.addWidget(self.main_app_widget)
        self.stacked_widget.setCurrentIndex(0)

    # ===============================================================
    # TELA DE INÍCIO E NAVEGAÇÃO
    # ===============================================================
    def build_start_screen(self):
        self.start_screen = QWidget()
        layout = QVBoxLayout(self.start_screen)
        
        lbl_title = QLabel("Plataforma de Aquisição e Análise Estrutural")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 26px; font-weight: bold; margin-bottom: 30px; margin-top: 50px;")
        layout.addWidget(lbl_title)
        
        lbl_sub = QLabel("Selecione o modo de operação desejado para iniciar:")
        lbl_sub.setAlignment(Qt.AlignCenter)
        lbl_sub.setStyleSheet("font-size: 16px; margin-bottom: 30px;")
        layout.addWidget(lbl_sub)
        
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(50, 0, 50, 0)
        
        btn_acq = QPushButton("Apenas Aquisição de Dados\n(Sinais no Tempo)")
        btn_acq.setMinimumHeight(120)
        btn_acq.setStyleSheet("font-size: 14px; background-color: #0366d6; color: white; font-weight: bold; border-radius: 8px;")
        btn_acq.clicked.connect(lambda: self.launch_app('acq'))
        
        btn_ema = QPushButton("Análise Modal Experimental\n(EMA - Com Martelo/Força)")
        btn_ema.setMinimumHeight(120)
        btn_ema.setStyleSheet("font-size: 14px; background-color: #2da44e; color: white; font-weight: bold; border-radius: 8px;")
        btn_ema.clicked.connect(lambda: self.launch_app('ema'))

        btn_oma = QPushButton("Análise Modal Operacional\n(OMA - Ruído Ambiente)")
        btn_oma.setMinimumHeight(120)
        btn_oma.setStyleSheet("font-size: 14px; background-color: #6f42c1; color: white; font-weight: bold; border-radius: 8px;")
        btn_oma.clicked.connect(lambda: self.launch_app('oma'))
        
        btn_layout.addWidget(btn_acq)
        btn_layout.addWidget(btn_ema)
        btn_layout.addWidget(btn_oma)
        layout.addLayout(btn_layout)
        layout.addStretch()

    def launch_app(self, mode):
        self.current_mode = mode
        self.tabs.clear()
        
        if mode == 'acq':
            self.tabs.addTab(self.tab_ch, "1. Mapeamento de Canais")
            self.tabs.addTab(self.tab_acq, "2. Aquisição de Dados")
            self.tabs.addTab(self.tab_post, "3. Pós-Processamento")
            
            self.acq_controls_widget.setVisible(True)
            self.ema_controls_widget.setVisible(False)
            self.oma_controls_widget.setVisible(False)
            self.container_pts.setVisible(True)
            self.container_oma_time.setVisible(False)
            
        elif mode == 'ema':
            self.tabs.addTab(self.tab_geom, "1. Geometria 3D")
            self.tabs.addTab(self.tab_ch, "2. Mapeamento de Canais")
            self.tabs.addTab(self.tab_acq, "3. Aquisição Modal (EMA)")
            self.tabs.addTab(self.tab_ema, "4. Análise pyEMA")
            
            self.acq_controls_widget.setVisible(False)
            self.ema_controls_widget.setVisible(True)
            self.oma_controls_widget.setVisible(False)
            self.container_pts.setVisible(True)
            self.container_oma_time.setVisible(False)
            self.btn_reject_ema.setEnabled(False)
            
        elif mode == 'oma':
            self.tabs.addTab(self.tab_geom, "1. Geometria 3D")
            self.tabs.addTab(self.tab_ch, "2. Mapeamento de Canais")
            self.tabs.addTab(self.tab_acq, "3. Aquisição Modal (OMA)")
            self.tabs.addTab(self.tab_post_oma, "4. Pós-Processamento OMA")
            
            self.acq_controls_widget.setVisible(False)
            self.ema_controls_widget.setVisible(False)
            self.oma_controls_widget.setVisible(True)
            self.container_pts.setVisible(False) 
            self.container_oma_time.setVisible(True) 
            
        self.reset_plots()
        self.tabs.setCurrentIndex(0)
        self.check_hardware_connection(manual=False)
        self.stacked_widget.setCurrentIndex(1)

    def return_to_home(self):
        if self.acq_thread and self.acq_thread.isRunning():
            self.stop_acquisition()
        self.stacked_widget.setCurrentIndex(0)

    # ===============================================================
    # CONSTRUÇÃO DA INTERFACE PRINCIPAL
    # ===============================================================
    def build_main_app(self):
        self.main_app_widget = QWidget()
        main_layout = QVBoxLayout(self.main_app_widget)
        
        global_frame = QFrame()
        global_frame.setMaximumHeight(60) 
        global_layout = QHBoxLayout(global_frame)
        global_layout.setContentsMargins(5, 5, 5, 5)
        
        btn_home = QPushButton("🏠 Voltar à Tela Inicial")
        btn_home.setStyleSheet("font-weight: bold; background-color: #6e7781; color: white;")
        btn_home.clicked.connect(self.return_to_home)
        global_layout.addWidget(btn_home)
        
        global_layout.addWidget(QLabel("Diretório:"))
        self.txt_dir = QLineEdit()
        self.txt_dir.setText(os.getcwd())
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
        
        # Criação de todas as abas
        self.tab_geom = QWidget(); self.setupTab_Geom()
        self.tab_ch = QWidget(); self.setupTab_Channels()
        self.tab_acq = QWidget(); self.setupTab_Acquisition()
        self.tab_post = QWidget(); self.setupTab_PostProcessing()
        self.tab_ema = QWidget(); self.setupTab_pyEMA()
        self.tab_post_oma = QWidget(); self.setupTab_PostProcessing_OMA()
        
        main_layout.addWidget(self.tabs)

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Selecionar Diretório de Trabalho", self.txt_dir.text())
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
                    try:
                        _ = dev.product_type 
                        if not dev.is_simulated:
                            active_devs.append(dev.name)
                    except: pass
                        
                if len(active_devs) > 0:
                    msg = f"✅ Hardware OK: {', '.join(active_devs)}"
                    connected = True
            except: pass
            
        if connected:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("background-color: #2da44e; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            if manual: QMessageBox.information(self, "Status de Hardware", f"Verificação concluída.\n{msg}")
        else:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("background-color: #cf222e; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            if manual: QMessageBox.warning(self, "Status de Hardware", "Nenhum hardware NI detectado fisicamente na porta USB.\nVerifique o cabo ou o NI MAX.")

    # ===============================================================
    # ABA 1: GEOMETRIA 3D
    # ===============================================================
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
        btn_add_node = QPushButton("+ Adicionar Nó")
        btn_add_node.clicked.connect(self.add_node_row)
        btn_del_node = QPushButton("- Excluir Nó")
        btn_del_node.clicked.connect(self.delete_node_row)
        node_btn_layout.addWidget(btn_add_node)
        node_btn_layout.addWidget(btn_del_node)

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
        btn_add_line = QPushButton("+ Adicionar Linha")
        btn_add_line.clicked.connect(self.add_line_row)
        btn_del_line = QPushButton("- Excluir Linha")
        btn_del_line.clicked.connect(self.delete_line_row)
        line_btn_layout.addWidget(btn_add_line)
        line_btn_layout.addWidget(btn_del_line)

        lay_lines.addWidget(self.table_lines)
        lay_lines.addLayout(line_btn_layout)
        group_lines.setLayout(lay_lines)
        left_panel.addWidget(group_lines)
        
        btn_geom_io_layout = QHBoxLayout()
        btn_load_geom = QPushButton("📂 Carregar Geometria (.json)")
        btn_load_geom.clicked.connect(self.load_geometry)
        btn_save_geom = QPushButton("💾 Salvar Geometria (.json)")
        btn_save_geom.clicked.connect(self.save_geometry)
        btn_geom_io_layout.addWidget(btn_load_geom)
        btn_geom_io_layout.addWidget(btn_save_geom)
        left_panel.addLayout(btn_geom_io_layout)
        
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

    def save_geometry(self):
        geom_data = {'nodes': [], 'lines': []}
        for r in range(self.table_nodes.rowCount()):
            try:
                nid = self.table_nodes.item(r, 0).text()
                x = self.table_nodes.item(r, 1).text()
                y = self.table_nodes.item(r, 2).text()
                z = self.table_nodes.item(r, 3).text()
                geom_data['nodes'].append({'id': nid, 'x': x, 'y': y, 'z': z})
            except: pass
            
        for r in range(self.table_lines.rowCount()):
            try:
                n1 = self.table_lines.item(r, 0).text()
                n2 = self.table_lines.item(r, 1).text()
                geom_data['lines'].append({'n1': n1, 'n2': n2})
            except: pass
            
        filename, _ = QFileDialog.getSaveFileName(self, "Salvar Geometria", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f: json.dump(geom_data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "Sucesso", "Geometria salva com sucesso!")
            except Exception as e: QMessageBox.critical(self, "Erro", f"Erro ao salvar arquivo:\n{str(e)}")

    def load_geometry(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Carregar Geometria", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f: geom_data = json.load(f)
                if 'nodes' in geom_data and 'lines' in geom_data:
                    self.table_nodes.setRowCount(0)
                    for node in geom_data['nodes']:
                        row = self.table_nodes.rowCount()
                        self.table_nodes.insertRow(row)
                        self.table_nodes.setItem(row, 0, QTableWidgetItem(str(node.get('id', ''))))
                        self.table_nodes.setItem(row, 1, QTableWidgetItem(str(node.get('x', '0'))))
                        self.table_nodes.setItem(row, 2, QTableWidgetItem(str(node.get('y', '0'))))
                        self.table_nodes.setItem(row, 3, QTableWidgetItem(str(node.get('z', '0'))))
                        
                    self.table_lines.setRowCount(0)
                    for line in geom_data['lines']:
                        row = self.table_lines.rowCount()
                        self.table_lines.insertRow(row)
                        self.table_lines.setItem(row, 0, QTableWidgetItem(str(line.get('n1', ''))))
                        self.table_lines.setItem(row, 1, QTableWidgetItem(str(line.get('n2', ''))))
                        
                    self.plot_geometry()
                    QMessageBox.information(self, "Sucesso", "Geometria carregada!")
                else: QMessageBox.warning(self, "Aviso", "Formato incompatível.")
            except Exception as e: QMessageBox.critical(self, "Erro", f"Erro ao carregar o arquivo:\n{str(e)}")

    def add_node_row(self):
        row = self.table_nodes.rowCount()
        self.table_nodes.insertRow(row)
        self.table_nodes.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.table_nodes.setItem(row, 1, QTableWidgetItem("0.0"))
        self.table_nodes.setItem(row, 2, QTableWidgetItem("0.0"))
        self.table_nodes.setItem(row, 3, QTableWidgetItem("0.0"))
        self.plot_geometry()

    def delete_node_row(self):
        selected = set(item.row() for item in self.table_nodes.selectedItems())
        if selected:
            for row in sorted(selected, reverse=True): self.table_nodes.removeRow(row)
        elif self.table_nodes.rowCount() > 0:
            self.table_nodes.removeRow(self.table_nodes.rowCount() - 1)
        self.plot_geometry()

    def add_line_row(self):
        row = self.table_lines.rowCount()
        self.table_lines.insertRow(row)
        self.table_lines.setItem(row, 0, QTableWidgetItem("1"))
        self.table_lines.setItem(row, 1, QTableWidgetItem("2"))
        self.plot_geometry()

    def delete_line_row(self):
        selected = set(item.row() for item in self.table_lines.selectedItems())
        if selected:
            for row in sorted(selected, reverse=True): self.table_lines.removeRow(row)
        elif self.table_lines.rowCount() > 0:
            self.table_lines.removeRow(self.table_lines.rowCount() - 1)
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
            max_dim = max(dx, dy, dz)
            try: ax.set_box_aspect((dx, max(dy, max_dim*0.3), max(dz, max_dim*0.3)))
            except AttributeError: pass
        
        ax.view_init(elev=20, azim=-60)
        self.fig_geom.tight_layout()
        self.canvas_geom.draw()

    # ===============================================================
    # ABA 2: MAPEAMENTO DE CANAIS
    # ===============================================================
    def setupTab_Channels(self):
        layout = QVBoxLayout(self.tab_ch)
        self.table_instruments = QTableWidget(0, 8)
        self.table_instruments.setHorizontalHeaderLabels(["Ativar", "Dispositivo/Canal", "Tipo", "Nó Físico", "Direção", "Sensibilidade", "Unidade", "IEPE"])
        self.table_instruments.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_instruments)
        
        ch_btn_layout = QHBoxLayout()
        btn_add_ch = QPushButton("+ Adicionar Canal")
        btn_add_ch.clicked.connect(lambda: self.add_instrument_row())
        btn_del_ch = QPushButton("- Remover Canal")
        btn_del_ch.clicked.connect(self.delete_instrument_row)
        ch_btn_layout.addWidget(btn_add_ch)
        ch_btn_layout.addWidget(btn_del_ch)
        layout.addLayout(ch_btn_layout)
        
        self.add_instrument_row({'device': 'cDAQ1Mod1', 'channel': 'ai0', 'type': 'Força', 'node': 1, 'dir': '+Z', 'sensitivity': 2.25, 'unit': 'mV/N', 'iepe': 'Ativado (2mA)'})
        self.add_instrument_row({'device': 'cDAQ1Mod1', 'channel': 'ai1', 'type': 'Aceleração', 'node': 2, 'dir': '+Z', 'sensitivity': 100.0, 'unit': 'mV/g', 'iepe': 'Ativado (2mA)'})
        
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Carregar Configuração (.json)")
        btn_load.clicked.connect(self.load_config)
        btn_save = QPushButton("Salvar Configuração (.json)")
        btn_save.clicked.connect(self.save_config)
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)
        
        btn_next = QPushButton("Avançar >>")
        btn_next.clicked.connect(lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1))
        layout.addWidget(btn_next)

    def add_instrument_row(self, ch=None):
        row = self.table_instruments.rowCount()
        self.table_instruments.insertRow(row)
        if ch is None: ch = {'device': 'cDAQ1Mod1', 'channel': f'ai{row}', 'enabled': True, 'type': 'Aceleração', 'node': 1, 'dir': '+Z', 'sensitivity': 100.0, 'unit': 'mV/g', 'iepe': 'Ativado (2mA)'}
            
        chk = QCheckBox(); chk.setChecked(ch.get('enabled', True)); chk.setStyleSheet("margin-left:50%;")
        self.table_instruments.setCellWidget(row, 0, chk)
        
        txt_ch = QLineEdit(f"{ch.get('device', 'cDAQ1Mod1')}/{ch.get('channel', 'ai0')}")
        self.table_instruments.setCellWidget(row, 1, txt_ch)
        
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
        
        # Removemos o sufixo fixo do spinbox pois agora existe um combobox ao lado
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
            dev = parts[0] if len(parts) > 0 else 'cDAQ1Mod1'
            chn = parts[1] if len(parts) > 1 else 'ai0'
            config.append({
                'enabled': self.table_instruments.cellWidget(i, 0).isChecked(), 'device': dev, 'channel': chn,
                'type': self.table_instruments.cellWidget(i, 2).currentText(), 'node': self.table_instruments.cellWidget(i, 3).value(),
                'dir': self.table_instruments.cellWidget(i, 4).currentText(), 'sensitivity': self.table_instruments.cellWidget(i, 5).value(),
                'unit': self.table_instruments.cellWidget(i, 6).currentText(),
                'iepe': self.table_instruments.cellWidget(i, 7).currentText()
            })
        filename, _ = QFileDialog.getSaveFileName(self, "Salvar Configuração", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f: json.dump(config, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "Sucesso", "Configurações salvas com sucesso!")
            except Exception as e: QMessageBox.critical(self, "Erro", f"Erro ao salvar arquivo:\n{str(e)}")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Carregar Configurações", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f: loaded = json.load(f)
                if isinstance(loaded, list):
                    self.table_instruments.setRowCount(0)
                    for ch in loaded: self.add_instrument_row(ch)
                    QMessageBox.information(self, "Sucesso", "Configurações carregadas!")
                else: QMessageBox.warning(self, "Aviso", "Formato incompatível.")
            except Exception as e: QMessageBox.critical(self, "Erro", f"Erro ao carregar o arquivo:\n{str(e)}")

    def get_active_channels(self):
        active = []
        for i in range(self.table_instruments.rowCount()):
            chk = self.table_instruments.cellWidget(i, 0)
            combo = self.table_instruments.cellWidget(i, 2)
            if chk and chk.isChecked() and combo.currentText() != "Desativado":
                parts = self.table_instruments.cellWidget(i, 1).text().split('/')
                dev = parts[0] if len(parts) > 0 else 'cDAQ1Mod1'
                chn = parts[1] if len(parts) > 1 else 'ai0'
                node_val = self.table_instruments.cellWidget(i, 3).value()
                dir_val = self.table_instruments.cellWidget(i, 4).currentText()
                tipo = combo.currentText()
                lbl = "Martelo" if tipo == 'Força' else f"Nó {node_val} ({dir_val})"
                
                active.append({
                    'device': dev, 'channel': chn, 'type': tipo, 'label': lbl, 
                    'sensitivity': self.table_instruments.cellWidget(i, 5).value(),
                    'unit': self.table_instruments.cellWidget(i, 6).currentText(),
                    'node': node_val, 'dir': dir_val,
                    'iepe': self.table_instruments.cellWidget(i, 7).currentText()
                })
        return active

    # ===============================================================
    # ABA 3: AQUISIÇÃO GERAL E EXPORTAÇÃO
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
        
        # Container ACQ/EMA (Pontos)
        self.container_pts = QWidget()
        layout_pts = QHBoxLayout(self.container_pts); layout_pts.setContentsMargins(0,0,0,0)
        layout_pts.addWidget(QLabel("Número de Pontos:"))
        self.c_pts_acq = QComboBox()
        self.c_pts_acq.addItems(["512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"])
        self.c_pts_acq.setEditable(True); self.c_pts_acq.setCurrentText("8192")
        layout_pts.addWidget(self.c_pts_acq)
        grid_timing.addWidget(self.container_pts, 1, 0, 1, 2)
        
        # Container OMA (Minutos)
        self.container_oma_time = QWidget()
        layout_oma_time = QHBoxLayout(self.container_oma_time); layout_oma_time.setContentsMargins(0,0,0,0)
        layout_oma_time.addWidget(QLabel("Duração da Medição (Minutos):"))
        self.spin_dur_oma = QDoubleSpinBox()
        self.spin_dur_oma.setRange(0.1, 1440.0); self.spin_dur_oma.setValue(1.0)
        layout_oma_time.addWidget(self.spin_dur_oma)
        grid_timing.addWidget(self.container_oma_time, 2, 0, 1, 2)

        self.lbl_timing_acq = QLabel()
        grid_timing.addWidget(self.lbl_timing_acq, 3, 0, 1, 2)

        group_timing.setLayout(grid_timing)
        left_panel.addWidget(group_timing)

        self.c_fs_acq.currentTextChanged.connect(self.update_timing_acq)
        self.c_pts_acq.currentTextChanged.connect(self.update_timing_acq)
        self.spin_dur_oma.valueChanged.connect(self.update_timing_acq)
        self.update_timing_acq()
        
        # MODO ACQ CONTROLS
        self.acq_controls_widget = QWidget()
        acq_layout = QVBoxLayout(self.acq_controls_widget)
        acq_layout.setContentsMargins(0,0,0,0)
        self.btn_monitor_acq = QPushButton("Monitorar Sinal Contínuo")
        self.btn_monitor_acq.setCheckable(True)
        self.btn_monitor_acq.setStyleSheet("background-color: #0366d6; color: white; min-height: 40px; font-weight: bold;")
        self.btn_monitor_acq.clicked.connect(lambda: self.toggle_monitor('acq'))
        acq_layout.addWidget(self.btn_monitor_acq)
        self.btn_record_acq = QPushButton("Gravar Sinal (.txt)")
        self.btn_record_acq.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_record_acq.clicked.connect(self.start_record)
        acq_layout.addWidget(self.btn_record_acq)
        left_panel.addWidget(self.acq_controls_widget)
        
        # MODO OMA CONTROLS
        self.oma_controls_widget = QWidget()
        oma_layout = QVBoxLayout(self.oma_controls_widget)
        oma_layout.setContentsMargins(0,0,0,0)
        self.chk_simulation_oma = QCheckBox("Modo Simulação (Gerar Ruído/Senos)")
        oma_layout.addWidget(self.chk_simulation_oma)
        
        self.group_modal_oma = QGroupBox("Controle de Malha de Teste OMA")
        grid_m_oma = QGridLayout()
        grid_m_oma.addWidget(QLabel("Qtd Total de Setups:"), 0, 0)
        self.spin_setups_oma = QSpinBox(); self.spin_setups_oma.setValue(3); self.spin_setups_oma.setMinimum(1)
        grid_m_oma.addWidget(self.spin_setups_oma, 0, 1)
        self.lbl_progress_oma = QLabel("Progresso: Aguardando")
        self.lbl_progress_oma.setStyleSheet("font-weight: bold; color: blue;")
        grid_m_oma.addWidget(self.lbl_progress_oma, 1, 0, 1, 2, Qt.AlignCenter)
        self.group_modal_oma.setLayout(grid_m_oma)
        oma_layout.addWidget(self.group_modal_oma)
        
        self.btn_monitor_oma = QPushButton("Monitorar Sinais OMA (Sem Gravar)")
        self.btn_monitor_oma.setCheckable(True)
        self.btn_monitor_oma.setStyleSheet("background-color: #d4a72c; color: black; min-height: 40px; font-weight: bold;")
        self.btn_monitor_oma.clicked.connect(lambda: self.toggle_monitor('oma'))
        oma_layout.addWidget(self.btn_monitor_oma)
        
        self.btn_record_oma = QPushButton("Iniciar Gravação Contínua OMA")
        self.btn_record_oma.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_record_oma.clicked.connect(self.start_oma_record)
        oma_layout.addWidget(self.btn_record_oma)
        left_panel.addWidget(self.oma_controls_widget)

        # MODO EMA CONTROLS
        self.ema_controls_widget = QWidget()
        ema_layout = QVBoxLayout(self.ema_controls_widget)
        ema_layout.setContentsMargins(0,0,0,0)
        
        self.group_trigger = QGroupBox("Trigger e Qualidade (Apenas EMA)")
        grid_trig = QGridLayout()
        grid_trig.addWidget(QLabel("Nível Trigger (N):"), 0, 0)
        self.spin_trig_lvl = QDoubleSpinBox()
        self.spin_trig_lvl.setRange(0.1, 1000.0)
        self.spin_trig_lvl.setValue(5.0)
        grid_trig.addWidget(self.spin_trig_lvl, 0, 1)
        self.chk_trigger = QCheckBox("Habilitar Gatilho (Trigger) no Martelo")
        self.chk_trigger.setChecked(True)
        grid_trig.addWidget(self.chk_trigger, 1, 0, 1, 2)
        self.chk_double_hit = QCheckBox("Detectar e Alertar Dupla Batida")
        self.chk_double_hit.setChecked(True)
        grid_trig.addWidget(self.chk_double_hit, 2, 0, 1, 2)
        self.group_trigger.setLayout(grid_trig)
        ema_layout.addWidget(self.group_trigger)
        
        self.group_modal = QGroupBox("Controle de Malha de Teste EMA")
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
        ema_layout.addWidget(self.group_modal)
        
        self.btn_start_ema = QPushButton("Armar Aquisição (Aguardar Impacto)")
        self.btn_start_ema.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_start_ema.clicked.connect(self.start_ema_acquisition)
        ema_layout.addWidget(self.btn_start_ema)
        
        self.btn_reject_ema = QPushButton("Rejeitar Último Impacto")
        self.btn_reject_ema.setStyleSheet("background-color: #d73a49; color: white; min-height: 30px; font-weight: bold;")
        self.btn_reject_ema.setEnabled(False)
        self.btn_reject_ema.clicked.connect(self.reject_last_impact)
        ema_layout.addWidget(self.btn_reject_ema)
        left_panel.addWidget(self.ema_controls_widget)
        
        # COMUM
        self.txt_export_name = QLineEdit("Teste_Estrutural")
        left_panel.addWidget(QLabel("Prefixo do Arquivo de Exportação:"))
        left_panel.addWidget(self.txt_export_name)
        
        self.lbl_export_status = QLabel("Aguardando...")
        self.lbl_export_status.setStyleSheet("color: gray; font-style: italic;")
        left_panel.addWidget(self.lbl_export_status)
        
        self.btn_stop = QPushButton("Parar Execução")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_acquisition)
        left_panel.addWidget(self.btn_stop)
        left_panel.addStretch()
        
        right_panel = QVBoxLayout()
        self.figure = Figure()
        self.canvas_sig = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas_sig)
        self.reset_plots()
        
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)

    def update_timing_ema(self):
        try:
            fs = float(self.c_fs_ema.currentText())
            pts = int(self.c_pts_ema.currentText())
            dur = pts / fs
            df = fs / pts
            self.lbl_timing_ema.setText(f"Duração do Bloco: {dur:.3f} s | Resolução (Δf): {df:.3f} Hz")
            self.lbl_timing_ema.setStyleSheet("font-weight: bold; color: #0366d6;")
        except ValueError:
            self.lbl_timing_ema.setText("Valores Inválidos!")
            self.lbl_timing_ema.setStyleSheet("font-weight: bold; color: red;")

    def update_timing_acq(self):
        try:
            fs = float(self.c_fs_acq.currentText())
            if self.current_mode == 'oma':
                dur_min = self.spin_dur_oma.value()
                pts = int(fs * dur_min * 60)
                self.lbl_timing_acq.setText(f"Amostras Totais: {pts}")
                self.lbl_timing_acq.setStyleSheet("font-weight: bold; color: #0366d6;")
            else:
                pts = int(self.c_pts_acq.currentText())
                dur = pts / fs
                df = fs / pts
                self.lbl_timing_acq.setText(f"Duração: {dur:.3f} s | Resolução (Δf): {df:.3f} Hz")
                self.lbl_timing_acq.setStyleSheet("font-weight: bold; color: #0366d6;")
        except ValueError:
            self.lbl_timing_acq.setText("Valores Inválidos!")
            self.lbl_timing_acq.setStyleSheet("font-weight: bold; color: red;")

    def reset_plots(self):
        self.figure.clf()
        if self.current_mode == 'acq' or self.current_mode == 'oma':
            self.ax_acq = self.figure.add_subplot(111)
            self.ax_acq.set_title("Respostas Acelerométricas")
            self.ax_acq.set_ylabel("Amplitude")
            self.ax_acq.set_xlabel("Tempo [s]")
            self.ax_acq.grid(True)
        else:
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
        if hasattr(self, 'canvas_sig'): self.canvas_sig.draw()

    # --- LÓGICA MODO ACQ E OMA ---
    def toggle_monitor(self, acq_type):
        btn = self.btn_monitor_acq if acq_type == 'acq' else self.btn_monitor_oma
        
        if btn.isChecked():
            active_channels = self.get_active_channels()
            if not active_channels:
                btn.setChecked(False)
                return QMessageBox.warning(self, "Erro", "Nenhum canal ativo selecionado!")
            for ch in active_channels:
                if ch['type'] == 'Força':
                    btn.setChecked(False)
                    return QMessageBox.warning(self, "Aviso", "Remova o canal do martelo neste modo.")
            try: 
                fs_val = float(self.c_fs_acq.currentText())
                if acq_type == 'acq': pts_val = int(self.c_pts_acq.currentText())
                else: pts_val = int(fs_val * self.spin_dur_oma.value() * 60)
            except ValueError: 
                btn.setChecked(False)
                return QMessageBox.warning(self, "Erro", "Verifique a taxa de amostragem.")

            config = {
                'fs': fs_val, 'nsamples': pts_val,
                'channels': active_channels, 'acq_type': f'{acq_type}_monitor',
                'simulation': self.chk_simulation_oma.isChecked() if acq_type == 'oma' else False
            }
            
            btn.setText("Parar Monitoramento")
            btn.setStyleSheet("background-color: #d73a49; color: white; min-height: 40px; font-weight: bold;")
            if acq_type == 'acq': self.btn_record_acq.setEnabled(False)
            else: self.btn_record_oma.setEnabled(False)
            
            self.monitor_buffer = None
            self.current_acq_type = f'{acq_type}_monitor'
            
            self.acq_thread = AcquisitionThread(config)
            self.acq_thread.data_signal.connect(self.route_data)
            self.acq_thread.error_signal.connect(self.handle_error)
            self.acq_thread.start()
        else:
            if self.acq_thread and self.acq_thread.isRunning():
                self.acq_thread.is_running = False
                self.acq_thread.wait(1000)
            
            if acq_type == 'acq':
                btn.setText("Monitorar Sinal Contínuo")
                btn.setStyleSheet("background-color: #0366d6; color: white; min-height: 40px; font-weight: bold;")
                self.btn_record_acq.setEnabled(True)
            else:
                btn.setText("Monitorar Sinais OMA")
                btn.setStyleSheet("background-color: #d4a72c; color: black; min-height: 40px; font-weight: bold;")
                self.btn_record_oma.setEnabled(True)

    def start_record(self):
        active_channels = self.get_active_channels()
        if not active_channels: return QMessageBox.warning(self, "Erro", "Nenhum canal ativo selecionado!")
        for ch in active_channels:
            if ch['type'] == 'Força': return QMessageBox.warning(self, "Aviso", "No modo 'Apenas Aquisição', mapeie apenas 'Aceleração'.")

        fs_val = float(self.c_fs_acq.currentText())
        pts_val = int(self.c_pts_acq.currentText())

        config = {
            'fs': fs_val, 'nsamples': pts_val,
            'channels': active_channels, 'acq_type': 'record'
        }
        
        self.btn_record_acq.setText("Adquirindo Dados...")
        self.btn_record_acq.setStyleSheet("background-color: #d4a72c; color: black; min-height: 40px; font-weight: bold;")
        self.btn_record_acq.setEnabled(False)
        self.btn_monitor_acq.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.current_acq_type = 'record'
        
        self.acq_thread = AcquisitionThread(config)
        self.acq_thread.data_signal.connect(self.route_data)
        self.acq_thread.error_signal.connect(self.handle_error)
        self.acq_thread.finished_signal.connect(self.acquisition_finished)
        self.acq_thread.start()

    def start_oma_record(self):
        active_channels = self.get_active_channels()
        if not active_channels: return QMessageBox.warning(self, "Erro", "Nenhum canal ativo selecionado!")
        
        fs_val = float(self.c_fs_acq.currentText())
        dur_min = self.spin_dur_oma.value()
        pts_val = int(fs_val * dur_min * 60)

        config = {
            'fs': fs_val, 'nsamples': pts_val, 'duration_min': dur_min,
            'channels': active_channels, 'acq_type': 'oma_record',
            'simulation': self.chk_simulation_oma.isChecked()
        }
        
        self.btn_record_oma.setText("Gravando Dados OMA...")
        self.btn_record_oma.setStyleSheet("background-color: #cf222e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_record_oma.setEnabled(False)
        self.btn_monitor_oma.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.current_acq_type = 'oma_record'
        
        self.acq_thread = AcquisitionThread(config)
        self.acq_thread.data_signal.connect(self.route_data)
        self.acq_thread.oma_progress_signal.connect(self.update_oma_progress)
        self.acq_thread.error_signal.connect(self.handle_error)
        self.acq_thread.oma_finished_signal.connect(self.oma_acquisition_finished)
        self.acq_thread.start()

    # --- LÓGICA MODO EMA ---
    def start_ema_acquisition(self):
        active_channels = self.get_active_channels()
        if not active_channels: return QMessageBox.warning(self, "Erro", "Nenhum canal ativo selecionado!")
        fs_val = float(self.c_fs_acq.currentText())
        pts_val = int(self.c_pts_acq.currentText())

        config = {
            'fs': fs_val, 'nsamples': pts_val,
            'channels': active_channels, 'acq_type': 'ema',
            'use_trigger': self.chk_trigger.isChecked(), 'trigger_level': self.spin_trig_lvl.value(),
            'check_double_hit': self.chk_double_hit.isChecked()
        }
        
        self.btn_start_ema.setText("Aguardando Impacto...")
        self.btn_start_ema.setStyleSheet("background-color: #d4a72c; color: black; min-height: 40px; font-weight: bold;")
        self.btn_start_ema.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.current_acq_type = 'ema'
        
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
        if 'oma' in self.current_acq_type: pass 
        else: self.acquisition_finished()

    def handle_error(self, err_msg):
        if hasattr(self, 'btn_monitor_acq') and self.btn_monitor_acq.isChecked(): self.toggle_monitor('acq')
        if hasattr(self, 'btn_monitor_oma') and self.btn_monitor_oma.isChecked(): self.toggle_monitor('oma')
        QMessageBox.critical(self, "Erro na Placa", f"Erro:\n{err_msg}")
        self.acquisition_finished()

    def acquisition_finished(self):
        if self.current_mode == 'acq':
            self.btn_record_acq.setText("Gravar Sinal (.txt)")
            self.btn_record_acq.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
            self.btn_record_acq.setEnabled(True)
            self.btn_monitor_acq.setEnabled(True)
        elif self.current_mode == 'ema':
            self.btn_start_ema.setText("Armar Aquisição (Aguardar Impacto)")
            self.btn_start_ema.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
            self.btn_start_ema.setEnabled(True)
            if len(self.node_data_buffer) > 0: self.btn_reject_ema.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def oma_acquisition_finished(self, full_data_matrix):
        self.btn_record_oma.setText("Iniciar Gravação Contínua OMA")
        self.btn_record_oma.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_record_oma.setEnabled(True)
        self.btn_monitor_oma.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if full_data_matrix.size > 0:
            active = self.get_active_channels()
            fs = float(self.c_fs_acq.currentText())
            timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
            base = self.txt_export_name.text()
            filename = f"{base}_Setup{self.current_node}_{timestamp}.txt"
            
            try:
                data_matrix_ms2 = full_data_matrix
                t = np.arange(data_matrix_ms2.shape[1]) / fs
                out = np.vstack((t.reshape(1,-1), data_matrix_ms2)).T
                labels = ["Tempo[s]"] + [f"{ch['label']} [{get_disp_unit(ch.get('unit', ''))}]" for ch in active]
                np.savetxt(filename, out, fmt="%.6e", delimiter='\t', header="\t".join(labels), comments='')
                self.lbl_export_status.setText(f"Salvo OMA: {filename}")
                self.lbl_export_status.setStyleSheet("color: green;")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Não gravou TXT:\n{e}")

            if self.current_node < self.spin_setups_oma.value():
                QMessageBox.information(self, "Setup OMA", f"Setup {self.current_node} concluído.\nMova sensores para o Setup {self.current_node+1}.")
                self.current_node += 1
            else:
                QMessageBox.information(self, "Fim", "Todos os Setups OMA concluídos!")
                self.current_node = 1
                
        self.lbl_progress_oma.setText(f"Setup {self.current_node}: Aguardando")

    def reject_last_impact(self):
        if not self.node_data_buffer: return
        self.node_data_buffer.pop() 
        self.current_impact = max(1, self.current_impact - 1)    
        self.lbl_progress.setText(f"Nó: {self.current_node}/{self.spin_nodes.value()} | Impacto: {self.current_impact}/{self.spin_impacts.value()}")
        self.reset_plots()
        self.ax1.text(0.5, 0.5, 'IMPACTO REJEITADO', transform=self.ax1.transAxes, color='red', fontsize=14, ha='center', weight='bold')
        self.canvas_sig.draw()
        if len(self.node_data_buffer) == 0: self.btn_reject_ema.setEnabled(False)

    # ===============================================================
    # ROTEADOR DE DADOS DA THREAD E PLOTAGEM GERAL
    # ===============================================================
    def route_data(self, time_axis, data_matrix, double_hit):
        if 'monitor' in self.current_acq_type:
            self.plot_monitor_acq(data_matrix)
        elif self.current_acq_type == 'record':
            self.plot_record_acq(time_axis, data_matrix)
        elif self.current_acq_type == 'oma_record':
            self.plot_monitor_acq(data_matrix) 
        elif self.current_acq_type == 'ema':
            self.plot_ema_data(time_axis, data_matrix, double_hit)

    def update_oma_progress(self, sec):
        total = self.spin_dur_oma.value() * 60.0
        self.lbl_progress_oma.setText(f"Setup {self.current_node}: {sec:.1f}s / {total:.1f}s")

    def plot_monitor_acq(self, data_matrix):
        fs = float(self.c_fs_acq.currentText())
        max_samps = int(fs * 1.5) 
        
        if self.monitor_buffer is None: self.monitor_buffer = data_matrix
        else:
            self.monitor_buffer = np.hstack((self.monitor_buffer, data_matrix))
            if self.monitor_buffer.shape[1] > max_samps: self.monitor_buffer = self.monitor_buffer[:, -max_samps:]
        
        t = np.linspace(0, self.monitor_buffer.shape[1]/fs, self.monitor_buffer.shape[1])
        self.ax_acq.clear()
        self.ax_acq.set_title("Resposta Vibratória no Tempo")
        self.ax_acq.set_ylabel("Amplitude")
        self.ax_acq.set_xlabel("Tempo [s]")
        self.ax_acq.grid(True)
        
        active = self.get_active_channels()
        step = max(1, int(fs / 1500)) 
        
        for idx, ch in enumerate(active):
            if idx >= self.monitor_buffer.shape[0]: break
            self.ax_acq.plot(t[::step], self.monitor_buffer[idx, ::step], label=f"{ch['label']} [{get_disp_unit(ch.get('unit', ''))}]")
            
        self.ax_acq.legend(loc="upper right")
        self.canvas_sig.draw_idle()

    # ===============================================================
    # COMUM: SALVAMENTO DE RESULTADOS MODAIS (EMA E OMA)
    # ===============================================================
    def save_modal_results(self):
        if self.current_mode == 'oma':
            if not hasattr(self, 'oma_results') or not self.oma_results['freqs']: 
                return QMessageBox.warning(self, "Aviso", "Sem resultados OMA para salvar.")
            frequencias_naturais = self.oma_results['freqs']
            amortecimentos = self.oma_results['zetas']
        else:
            if not hasattr(self, 'acc') or not hasattr(self, 'A_shapes'): 
                return QMessageBox.warning(self, "Aviso", "Sem resultados EMA para salvar.")
            frequencias_naturais = self.acc.nat_freq
            amortecimentos = self.acc.nat_xi
            
        filename, _ = QFileDialog.getSaveFileName(self, "Salvar Resultados Modais", "formas_modais.txt", "Text Files (*.txt)")
        if not filename: return
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Modo\tFrequencia(Hz)\tAmortecimento(%)\n")
                for i in range(len(frequencias_naturais)):
                    f.write(f"{i+1}\t{frequencias_naturais[i]:.4f}\t{amortecimentos[i]*100:.4f}\n")
            QMessageBox.information(self, "Sucesso", f"Resultados salvos com sucesso em:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao salvar arquivo de texto:\n{str(e)}")

    # ===============================================================
    # ABA 3 (MODO ACQ ONLY): PÓS-PROCESSAMENTO / LEITOR .TXT
    # ===============================================================
    def setupTab_PostProcessing(self):
        layout = QHBoxLayout(self.tab_post)
        left_panel = QVBoxLayout()
        
        group_file = QGroupBox("Gerenciamento de Arquivos")
        file_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_load_txt = QPushButton("📂 Abrir Arquivo (.txt)")
        self.btn_load_txt.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold; min-height: 35px;")
        self.btn_load_txt.clicked.connect(self.load_external_txt)
        
        self.btn_load_last = QPushButton("🔄 Última Medição")
        self.btn_load_last.setStyleSheet("background-color: #2da44e; color: white; font-weight: bold; min-height: 35px;")
        self.btn_load_last.clicked.connect(self.load_last_txt)
        
        btn_layout.addWidget(self.btn_load_txt)
        btn_layout.addWidget(self.btn_load_last)
        file_layout.addLayout(btn_layout)
        
        self.lbl_loaded_file = QLabel("Nenhum arquivo carregado.")
        self.lbl_loaded_file.setAlignment(Qt.AlignCenter)
        self.lbl_loaded_file.setStyleSheet("color: gray; font-style: italic; margin-top: 5px;")
        file_layout.addWidget(self.lbl_loaded_file)
        
        group_file.setLayout(file_layout)
        left_panel.addWidget(group_file)
        
        group_stats = QGroupBox("Estatísticas do Sinal (Pós-Processamento)")
        stats_layout = QVBoxLayout()
        
        self.table_stats = QTableWidget(0, 5)
        self.table_stats.setHorizontalHeaderLabels(["Canal", "Pico", "RMS", "Crista", "F. Pico [Hz]"])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_stats.setEditTriggers(QTableWidget.NoEditTriggers) 
        self.table_stats.setSelectionBehavior(QTableWidget.SelectRows) 
        stats_layout.addWidget(self.table_stats)
        
        group_stats.setLayout(stats_layout)
        left_panel.addWidget(group_stats)
        
        right_panel = QVBoxLayout()
        self.fig_post = Figure()
        self.ax_post_time = self.fig_post.add_subplot(211)
        self.ax_post_fft = self.fig_post.add_subplot(212)
        self.canvas_post = FigureCanvas(self.fig_post)
        right_panel.addWidget(self.canvas_post)
        
        self.reset_post_plots() 
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)

    def reset_post_plots(self):
        self.ax_post_time.clear()
        self.ax_post_time.set_title("Sinal no Tempo")
        self.ax_post_time.set_ylabel("Amplitude")
        self.ax_post_time.set_xlabel("Tempo [s]")
        self.ax_post_time.grid(True)

        self.ax_post_fft.clear()
        self.ax_post_fft.set_title("Espectro de Frequência (FFT)")
        self.ax_post_fft.set_ylabel("Amplitude")
        self.ax_post_fft.set_xlabel("Frequência [Hz]")
        self.ax_post_fft.grid(True)
        
        self.fig_post.tight_layout()
        if hasattr(self, 'canvas_post'):
            self.canvas_post.draw()

    def load_external_txt(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Arquivo Temporal", "", "Text Files (*.txt)")
        if filename:
            self.process_txt_file(filename)

    def load_last_txt(self):
        if self.last_saved_txt and os.path.exists(self.last_saved_txt):
            self.process_txt_file(self.last_saved_txt)
        else:
            QMessageBox.warning(self, "Aviso", "Nenhum arquivo recente foi salvo ainda nesta sessão.")

    def process_txt_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                
            labels = header.split('\t')[1:]
            if not labels: labels = ["Canal 1"]
            
            data = np.loadtxt(filepath, delimiter='\t', skiprows=1)
            time_axis = data[:, 0]
            data_matrix = data[:, 1:].T
            if len(data_matrix.shape) == 1:
                data_matrix = data_matrix.reshape(1, -1)
                
            fs = 1.0 / (time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 1000.0
            
            self.process_and_plot_post(time_axis, data_matrix, labels, fs)
            
            self.lbl_loaded_file.setText(f"Arquivo carregado: {os.path.basename(filepath)}")
            self.lbl_loaded_file.setStyleSheet("color: green; font-weight: bold; margin-top: 5px;")
        except Exception as e:
            QMessageBox.critical(self, "Erro de Leitura", f"Falha ao ler o arquivo:\n{str(e)}")

    def process_and_plot_post(self, time_axis, data_matrix, labels, fs):
        self.reset_post_plots()
        self.table_stats.setRowCount(0)
        
        for idx in range(data_matrix.shape[0]):
            sig = data_matrix[idx, :]
            lbl = labels[idx] if idx < len(labels) else f"Canal {idx+1}"
            
            self.ax_post_time.plot(time_axis, sig, alpha=0.8, label=lbl)
            
            N = len(sig)
            yf = np.fft.rfft(sig)
            xf = np.fft.rfftfreq(N, d=1/fs)
            amp = 2.0/N * np.abs(yf)
            self.ax_post_fft.plot(xf, amp, alpha=0.8, label=lbl)
            
            rms = np.sqrt(np.mean(sig**2))
            peak = np.max(np.abs(sig))
            crest = peak / rms if rms > 0 else 0
            peak_freq = xf[np.argmax(amp)]
            
            row = self.table_stats.rowCount()
            self.table_stats.insertRow(row)
            self.table_stats.setItem(row, 0, QTableWidgetItem(lbl))
            self.table_stats.setItem(row, 1, QTableWidgetItem(f"{peak:.4f}"))
            self.table_stats.setItem(row, 2, QTableWidgetItem(f"{rms:.4f}"))
            self.table_stats.setItem(row, 3, QTableWidgetItem(f"{crest:.2f}"))
            self.table_stats.setItem(row, 4, QTableWidgetItem(f"{peak_freq:.1f}"))
            
        self.ax_post_time.legend(loc="upper right")
        self.ax_post_fft.legend(loc="upper right")
        self.fig_post.tight_layout()
        self.canvas_post.draw()

    # ===============================================================
    # ABA 4 (EMA): ANÁLISE MODAL EXPERIMENTAL (PYEMA)
    # ===============================================================
    def setupTab_pyEMA(self):
        layout = QHBoxLayout(self.tab_ema)
        left_panel = QVBoxLayout()
        
        self.list_uffs = QTableWidget(0, 1)
        self.list_uffs.setHorizontalHeaderLabels(["Arquivos .uff"])
        self.list_uffs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left_panel.addWidget(self.list_uffs)
        
        btn_add = QPushButton("+ Adicionar UFFs")
        btn_add.clicked.connect(self.load_uffs)
        left_panel.addWidget(btn_add)
        
        btn_clear_analysis = QPushButton("Limpar Dados / Resetar")
        btn_clear_analysis.setStyleSheet("background-color: #d73a49; color: white; font-weight: bold;")
        btn_clear_analysis.clicked.connect(self.clear_analysis_data)
        left_panel.addWidget(btn_clear_analysis)
        
        g_param = QGroupBox("Parâmetros de Cálculo da FRF (H1) e Ajuste")
        gp = QGridLayout()
        
        gp.addWidget(QLabel("Taxa de Amostragem (Hz):"), 0, 0)
        self.c_fs_ema = QComboBox()
        self.c_fs_ema.addItems(["1024", "2048", "4096", "8192", "16384"])
        self.c_fs_ema.setEditable(True)
        self.c_fs_ema.setCurrentText("4096")
        gp.addWidget(self.c_fs_ema, 0, 1)

        gp.addWidget(QLabel("Número de Pontos (H1):"), 1, 0)
        self.c_pts_ema = QComboBox()
        self.c_pts_ema.addItems(["512", "1024", "2048", "4096", "8192"])
        self.c_pts_ema.setEditable(True)
        self.c_pts_ema.setCurrentText("2048")
        gp.addWidget(self.c_pts_ema, 1, 1)

        self.lbl_timing_ema = QLabel()
        gp.addWidget(self.lbl_timing_ema, 2, 0, 1, 2)
        
        self.c_fs_ema.currentTextChanged.connect(self.update_timing_ema)
        self.c_pts_ema.currentTextChanged.connect(self.update_timing_ema)
        self.update_timing_ema()
        
        gp.addWidget(QLabel("Mapeamento Modal 3D:"), 3, 0, 1, 2)
        self.c_map_base = QComboBox()
        self.c_map_base.addItems([
            "Auto (Detectar pelo arquivo UFF)", 
            "Nós de Resposta (Ex: Acelerômetros Móveis)", 
            "Nós de Referência (Ex: Martelo Móvel)"
        ])
        gp.addWidget(self.c_map_base, 3, 2)

        gp.addWidget(QLabel("Pré-Trigger (pontos):"), 4, 0)
        self.s_pretrig = QSpinBox()
        self.s_pretrig.setRange(0, 500)
        self.s_pretrig.setValue(1)
        gp.addWidget(self.s_pretrig, 4, 1)

        gp.addWidget(QLabel("Freq. Mín/Máx (Hz):"), 5, 0)
        self.sp_fmin = QSpinBox(); self.sp_fmin.setRange(1, 10000); self.sp_fmin.setValue(10); gp.addWidget(self.sp_fmin, 5, 1)
        self.sp_fmax = QSpinBox(); self.sp_fmax.setRange(10, 20000); self.sp_fmax.setValue(600); gp.addWidget(self.sp_fmax, 5, 2)
        
        gp.addWidget(QLabel("Janelamento (Window):"), 6, 0)
        self.c_window = QComboBox()
        self.c_window.addItems(["Retangular (Nenhum)", "Exponencial", "Hanning"])
        gp.addWidget(self.c_window, 6, 1, 1, 2)

        g_param.setLayout(gp)
        left_panel.addWidget(g_param)
        
        self.btn_run_ema = QPushButton("Extrair Modos (Abre pyEMA)")
        self.btn_run_ema.setStyleSheet("background-color: #2da44e; color: white; min-height: 40px; font-weight: bold;")
        self.btn_run_ema.clicked.connect(self.run_ema)
        left_panel.addWidget(self.btn_run_ema)

        self.btn_save_res_ema = QPushButton("Salvar Resultados (.txt)")
        self.btn_save_res_ema.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold; min-height: 35px;")
        self.btn_save_res_ema.setEnabled(False)
        self.btn_save_res_ema.clicked.connect(self.save_modal_results)
        left_panel.addWidget(self.btn_save_res_ema)
        
        g_visual = QGroupBox("Controles de Visualização 3D")
        gv = QGridLayout()
        gv.addWidget(QLabel("Direção da Deformação:"), 0, 0)
        self.c_eixo = QComboBox()
        self.c_eixo.addItems(["Auto (Ler arquivo UFF)", "Forçar Bending em Z (Vertical)", "Forçar Bending em Y (Lateral)", "Forçar Bending em X (Axial)"])
        self.c_eixo.setCurrentIndex(1)
        self.c_eixo.currentIndexChanged.connect(self.plot_3d_mode)
        gv.addWidget(self.c_eixo, 0, 1)

        gv.addWidget(QLabel("Multiplicador de Escala:"), 1, 0)
        self.s_escala = QDoubleSpinBox()
        self.s_escala.setRange(0.1, 20.0)
        self.s_escala.setValue(1.0)
        self.s_escala.setSingleStep(0.5)
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

    def load_uffs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Selecionar UFF", "", "UFF (*.uff)")
        for f in files:
            r = self.list_uffs.rowCount()
            self.list_uffs.insertRow(r)
            self.list_uffs.setItem(r, 0, QTableWidgetItem(f))

    def clear_analysis_data(self):
        if self.current_mode == 'oma':
            self.lbl_loaded_oma.setText("Nenhum arquivo carregado.")
            self.oma_data_raw = None
            self.oma_psd_mat = None
            self.table_oma_res.setRowCount(0)
            self.c_oma_view_mode.clear()
            self.c_oma_view_mode.addItem("Selecione o Modo para Visualizar")
            self.reset_oma_plots()
        else:
            self.list_uffs.setRowCount(0)
            self.c_view_mode.clear()
            self.c_view_mode.addItem("Selecione o Modo após a Análise")
            if hasattr(self, 'A_shapes'): del self.A_shapes
            if hasattr(self, 'acc'): del self.acc
            self.btn_save_res_ema.setEnabled(False)
            self.fig_ema.clf()
            self.canvas_ema.draw()

    def run_ema(self):
        if not HAS_PYEMA: return QMessageBox.warning(self, "Aviso", "Módulo pyEMA não encontrado.")
            
        arquivos = [self.list_uffs.item(i, 0).text() for i in range(self.list_uffs.rowCount())]
        if not arquivos: return QMessageBox.warning(self, "Aviso", "Adicione arquivos .uff primeiro.")

        try:
            tamanho_bloco = int(self.c_pts_ema.currentText())
            fs_manual = float(self.c_fs_ema.currentText())
        except ValueError:
            return QMessageBox.warning(self, "Erro", "Taxa de amostragem ou número de pontos inválidos.")

        pre_trigger = self.s_pretrig.value()
        tipo_janela = self.c_window.currentText()

        if "Exponencial" in tipo_janela: win = np.exp(-4.6 * np.linspace(0, 1, tamanho_bloco))
        elif "Hanning" in tipo_janela: win = np.hanning(tamanho_bloco)
        else: win = np.ones(tamanho_bloco)

        todas_frfs = []
        self.frf_dofs = []
        frequencias = np.fft.rfftfreq(tamanho_bloco, d=1/fs_manual)
        
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
                            f_seg = forca[inicio:fim]
                            a_seg = aceleracao[inicio:fim] * win
                            
                            F = np.fft.rfft(f_seg)
                            A = np.fft.rfft(a_seg)
                            Sxx_soma += F * np.conj(F)
                            Sxy_soma += A * np.conj(F)
                            num_blocos += 1
                            
                    if num_blocos > 0:
                        Gxx = np.real(Sxx_soma / num_blocos)
                        Gxx[Gxx == 0] = 1e-12 
                        todas_frfs.append((Sxy_soma / num_blocos) / Gxx)
                        self.frf_dofs.append({
                            'rsp_node': ds_acc['rsp_node'], 'rsp_dir': ds_acc['rsp_dir'], 
                            'ref_node': ds_acc['ref_node'], 'ref_dir': ds_acc['ref_dir']
                        })
                        
            if not todas_frfs: return QMessageBox.warning(self, "Aviso", "Nenhum bloco válido encontrado para cálculo da FRF.")

            tamanho_minimo = min(len(frf) for frf in todas_frfs)
            todas_frfs = [frf[:tamanho_minimo] for frf in todas_frfs]
            frequencias = frequencias[:tamanho_minimo]

            self.acc = pyEMA.Model(frf=np.array(todas_frfs), freq=frequencias, lower=self.sp_fmin.value(), upper=self.sp_fmax.value(), pol_order_high=60)
            self.acc.get_poles()
            
            plt.show = matplotlib.pyplot.show
            QMessageBox.information(self, "Seleção de Polos", "A janela interativa do pyEMA vai abrir.\nSelecione os polos (X verdes) e feche a janela.")
            self.acc.select_poles()
            plt.show = lambda *args, **kwargs: None
            
            if not hasattr(self.acc, 'nat_freq') or len(self.acc.nat_freq) == 0: return
            self.H_synth, self.A_shapes = self.acc.get_constants(whose_poles='own')
            
            self.c_view_mode.blockSignals(True)
            self.c_view_mode.clear()
            for i, (fn, zeta) in enumerate(zip(self.acc.nat_freq, self.acc.nat_xi)):
                self.c_view_mode.addItem(f"Modo {i+1}: {fn:.2f} Hz, amort: {zeta*100:.2f}%")
            self.c_view_mode.blockSignals(False)
            
            self.btn_save_res_ema.setEnabled(True)
            self.plot_3d_mode()

        except Exception as e:
            QMessageBox.critical(self, "Erro no Processamento", f"Ocorreu um erro ao calcular as FRFs:\n{str(e)}")

    # ===============================================================
    # ABA 5 (OMA ONLY): PÓS-PROCESSAMENTO OMA (MODALYZER)
    # ===============================================================
    def setupTab_PostProcessing_OMA(self):
        layout = QHBoxLayout(self.tab_post_oma)
        left_panel = QVBoxLayout()
        
        # --- CARREGAR DADOS ---
        group_file = QGroupBox("Gerenciamento de Arquivos")
        file_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.btn_load_oma_txt = QPushButton("📂 Abrir Arquivo (.txt)")
        self.btn_load_oma_txt.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold; min-height: 35px;")
        self.btn_load_oma_txt.clicked.connect(self.load_oma_txt)
        btn_layout.addWidget(self.btn_load_oma_txt)
        file_layout.addLayout(btn_layout)
        self.lbl_loaded_oma = QLabel("Nenhum arquivo carregado.")
        self.lbl_loaded_oma.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self.lbl_loaded_oma)
        group_file.setLayout(file_layout)
        left_panel.addWidget(group_file)

        # --- PRE-PROCESSAMENTO E PSD ---
        group_pre = QGroupBox("1. Pré-Processamento e Inspeção (SVD)")
        pre_layout = QGridLayout()
        pre_layout.addWidget(QLabel("Fator de Dizimação:"), 0, 0)
        self.s_decimation = QSpinBox()
        self.s_decimation.setRange(1, 100)
        self.s_decimation.setValue(1)
        pre_layout.addWidget(self.s_decimation, 0, 1)
        
        self.chk_oma_detrend = QCheckBox("Aplicar Detrend (Remover Tendência Linear)")
        self.chk_oma_detrend.setChecked(True)
        pre_layout.addWidget(self.chk_oma_detrend, 1, 0, 1, 2)
        
        self.btn_calc_psd = QPushButton("📊 Gerar Espectro SVD (PSD)")
        self.btn_calc_psd.setStyleSheet("background-color: #d4a72c; font-weight: bold;")
        self.btn_calc_psd.clicked.connect(self.calc_oma_psd)
        pre_layout.addWidget(self.btn_calc_psd, 2, 0, 1, 2)
        group_pre.setLayout(pre_layout)
        left_panel.addWidget(group_pre)

        # --- EXTRAÇÃO OMA ---
        group_oma = QGroupBox("2. Extração Modal OMA")
        oma_layout = QGridLayout()
        oma_layout.addWidget(QLabel("Método:"), 0, 0)
        self.c_oma_method = QComboBox()
        self.c_oma_method.addItems(["EFDD (Enhanced Frequency Domain Decomposition)", "SSI (Stochastic Subspace Identification)"])
        oma_layout.addWidget(self.c_oma_method, 0, 1)

        oma_layout.addWidget(QLabel("Frequências Alvo (Hz):"), 1, 0)
        self.txt_oma_freqs = QLineEdit("21, 27, 190, 590")
        oma_layout.addWidget(self.txt_oma_freqs, 1, 1)
        
        self.btn_run_oma = QPushButton("⚙️ Extrair Parâmetros Modais")
        self.btn_run_oma.setStyleSheet("background-color: #2da44e; color: white; font-weight: bold; min-height: 35px;")
        self.btn_run_oma.clicked.connect(self.run_oma_extraction)
        oma_layout.addWidget(self.btn_run_oma, 2, 0, 1, 2)
        group_oma.setLayout(oma_layout)
        left_panel.addWidget(group_oma)

        # --- RESULTADOS ---
        group_res = QGroupBox("3. Resultados e Formas Modais")
        res_layout = QVBoxLayout()
        self.table_oma_res = QTableWidget(0, 3)
        self.table_oma_res.setHorizontalHeaderLabels(["Modo", "Frequência [Hz]", "Amortecimento [%]"])
        self.table_oma_res.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        res_layout.addWidget(self.table_oma_res)
        
        gv_oma = QGridLayout()
        gv_oma.addWidget(QLabel("Direção da Deformação:"), 0, 0)
        self.c_oma_eixo = QComboBox()
        self.c_oma_eixo.addItems(["Auto", "Forçar Bending em Z", "Forçar Bending em Y", "Forçar Bending em X"])
        self.c_oma_eixo.setCurrentIndex(1)
        self.c_oma_eixo.currentIndexChanged.connect(self.plot_3d_mode)
        gv_oma.addWidget(self.c_oma_eixo, 0, 1)

        gv_oma.addWidget(QLabel("Escala:"), 1, 0)
        self.s_oma_escala = QDoubleSpinBox()
        self.s_oma_escala.setRange(0.1, 20.0)
        self.s_oma_escala.setValue(1.0)
        self.s_oma_escala.setSingleStep(0.5)
        self.s_oma_escala.valueChanged.connect(self.plot_3d_mode)
        gv_oma.addWidget(self.s_oma_escala, 1, 1)
        res_layout.addLayout(gv_oma)
        
        self.c_oma_view_mode = QComboBox()
        self.c_oma_view_mode.addItem("Selecione o Modo para Visualizar")
        self.c_oma_view_mode.currentIndexChanged.connect(self.plot_3d_mode)
        res_layout.addWidget(QLabel("Visualizar Forma Modal 3D:"))
        res_layout.addWidget(self.c_oma_view_mode)
        
        self.btn_save_res_oma = QPushButton("Salvar Resultados OMA (.txt)")
        self.btn_save_res_oma.setStyleSheet("background-color: #0366d6; color: white; font-weight: bold; min-height: 30px;")
        self.btn_save_res_oma.setEnabled(False)
        self.btn_save_res_oma.clicked.connect(self.save_modal_results)
        res_layout.addWidget(self.btn_save_res_oma)

        group_res.setLayout(res_layout)
        left_panel.addWidget(group_res)
        left_panel.addStretch()
        
        # --- GRÁFICOS DIRETOS ---
        right_panel = QVBoxLayout()
        self.tabs_oma_plots = QTabWidget()
        
        self.tab_oma_sig = QWidget()
        sig_layout = QVBoxLayout(self.tab_oma_sig)
        self.fig_oma_sig = Figure()
        self.ax_oma_time = self.fig_oma_sig.add_subplot(211)
        self.ax_oma_svd = self.fig_oma_sig.add_subplot(212)
        self.canvas_oma_sig = FigureCanvas(self.fig_oma_sig)
        sig_layout.addWidget(self.canvas_oma_sig)
        self.tabs_oma_plots.addTab(self.tab_oma_sig, "Sinais e Espectro SVD")
        
        self.tab_oma_3d = QWidget()
        lay_3d = QVBoxLayout(self.tab_oma_3d)
        self.fig_oma_3d = Figure()
        self.canvas_oma_3d = FigureCanvas(self.fig_oma_3d)
        lay_3d.addWidget(self.canvas_oma_3d)
        self.tabs_oma_plots.addTab(self.tab_oma_3d, "Forma Modal 3D")
        
        right_panel.addWidget(self.tabs_oma_plots)
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)
        
        self.reset_oma_plots()

    def reset_oma_plots(self):
        self.ax_oma_time.clear()
        self.ax_oma_time.set_title("Sinal no Tempo")
        self.ax_oma_time.grid(True)
        self.ax_oma_svd.clear()
        self.ax_oma_svd.set_title("Valores Singulares (SVD do PSD)")
        self.ax_oma_svd.set_xlabel("Frequência [Hz]")
        self.ax_oma_svd.grid(True)
        self.fig_oma_sig.tight_layout()
        if hasattr(self, 'canvas_oma_sig'): self.canvas_oma_sig.draw()

    def load_oma_txt(self):
        if not HAS_MODALYZER:
            return QMessageBox.warning(self, "Erro", "As bibliotecas 'modalyzer', 'torch' ou 'pandas' não estão instaladas.")
            
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Arquivo OMA (.txt)", "", "Text Files (*.txt)")
        if not filename: return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                
            self.oma_labels = header.split('\t')[1:]
            
            self.oma_dofs_parsed = []
            for lbl in self.oma_labels:
                match = re.search(r'Nó\s+(\d+)\s*\(\s*([+-][XYZ])\s*\)', lbl)
                if match:
                    nid = int(match.group(1))
                    dir_str = match.group(2)
                    dir_map = {'+X': 1, '-X': -1, '+Y': 2, '-Y': -2, '+Z': 3, '-Z': -3}
                    self.oma_dofs_parsed.append({'rsp_node': nid, 'rsp_dir': dir_map.get(dir_str, 3)})
                else:
                    self.oma_dofs_parsed.append({'rsp_node': 1, 'rsp_dir': 3})

            df = pd.read_csv(filename, sep=r'\s+', skiprows=1, header=None)
            self.oma_time_raw = df.iloc[:, 0].values
            self.oma_data_raw = df.iloc[:, 1:].values
            
            dt = self.oma_time_raw[1] - self.oma_time_raw[0]
            self.oma_fs = 1.0 / dt
            
            self.ax_oma_time.clear()
            self.ax_oma_time.plot(self.oma_time_raw, self.oma_data_raw, alpha=0.7)
            self.ax_oma_time.set_title("Sinal no Tempo Original")
            self.fig_oma_sig.tight_layout()
            self.canvas_oma_sig.draw()
            
            self.lbl_loaded_oma.setText(f"Carregado: {os.path.basename(filename)}")
            self.lbl_loaded_oma.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao ler arquivo: {str(e)}")

    def calc_oma_psd(self):
        if self.oma_data_raw is None:
            return QMessageBox.warning(self, "Erro", "Carregue um arquivo primeiro.")
            
        try:
            dec = self.s_decimation.value()
            do_detrend = self.chk_oma_detrend.isChecked()
            
            dados_brutos = np.nan_to_num(self.oma_data_raw, nan=0.0).astype(np.float64)
            tempos_brutos = np.nan_to_num(self.oma_time_raw, nan=0.0).astype(np.float64)
            
            dados_limpos, t_limpo = preprocess(dados_brutos, tempos_brutos, do_detrend=do_detrend, decimate_downsample=dec)
            
            dados_limpos = dados_limpos - np.mean(dados_limpos, axis=0)
            dados_limpos += np.random.normal(0, 1e-10, dados_limpos.shape)
            
            self.oma_dt_novo = t_limpo[1] - t_limpo[0]
            self.oma_dados_efdd = np.ascontiguousarray(dados_limpos.T, dtype=np.float64) 
            
            try:
                self.oma_psd_mat, self.oma_f_psd, self.oma_ss1, self.oma_UU, self.oma_t_out = FDD_SVD_Diagram(self.oma_dados_efdd, self.oma_dt_novo)
            except Exception as inner_e:
                if "lt_cpu" in str(inner_e) or "ComplexFloat" in str(inner_e):
                    self.oma_psd_mat, self.oma_f_psd = psd_func(self.oma_dados_efdd, self.oma_dt_novo, show_semilog=False, nseg=256, pov=0.5)
                    self.oma_ss1, self.oma_UU = svd_psd(self.oma_psd_mat, self.oma_f_psd, show_semilog=False)
                    self.oma_t_out = np.arange(len(self.oma_f_psd)) * self.oma_dt_novo 
                else:
                    raise inner_e
                    
            plt.close('all') 
            
            self.ax_oma_svd.clear()
            
            if isinstance(self.oma_ss1, torch.Tensor):
                ss1_np = self.oma_ss1.detach().cpu().numpy()
            else:
                ss1_np = np.array(self.oma_ss1)
                
            ss1_np = np.abs(ss1_np)
            
            if ss1_np.ndim == 1:
                sv1 = ss1_np
                sv2 = None
            elif ss1_np.shape[0] == len(self.oma_f_psd):
                sv1 = ss1_np[:, 0]
                sv2 = ss1_np[:, 1] if ss1_np.shape[1] > 1 else None
            else:
                sv1 = ss1_np[0, :]
                sv2 = ss1_np[1, :] if ss1_np.shape[0] > 1 else None

            self.ax_oma_svd.plot(self.oma_f_psd, sv1, color='blue', label='SV 1')
            if sv2 is not None:
                self.ax_oma_svd.plot(self.oma_f_psd, sv2, color='red', alpha=0.5, label='SV 2')
                
            self.ax_oma_svd.set_yscale('log')
            self.ax_oma_svd.set_title("Espectro SVD (Identifique os picos visualmente)")
            self.ax_oma_svd.set_xlabel("Frequência [Hz]")
            self.ax_oma_svd.set_xlim(0, self.oma_f_psd[-1])
            self.ax_oma_svd.legend()
            self.fig_oma_sig.tight_layout()
            self.canvas_oma_sig.draw()
            QMessageBox.information(self, "SVD", "Espectro SVD gerado! Verifique os picos no gráfico inferior.")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro Modalyzer", f"Erro no cálculo do SVD:\n{str(e)}")

    def run_oma_extraction(self):
        if self.oma_psd_mat is None or self.oma_dados_efdd is None:
            return QMessageBox.warning(self, "Erro", "Gere o Espectro SVD primeiro.")
            
        freqs_str = self.txt_oma_freqs.text()
        try:
            target_freqs = [float(f.strip()) for f in freqs_str.split(',') if f.strip()]
        except:
            return QMessageBox.warning(self, "Erro", "Frequências devem ser separadas por vírgula (ex: 21, 27, 190).")

        method = self.c_oma_method.currentText()
        self.oma_results = {'freqs': [], 'zetas': [], 'modes': []}
        self.table_oma_res.setRowCount(0)
        
        try:
            if "EFDD" in method:
                nat_freqs_input = [target_freqs] 
                EFDD_Mode_shapes, SDOF, t_sdof = FDD_ModeShape(
                    nat_freqs_input, self.oma_psd_mat, self.oma_f_psd, self.oma_ss1, self.oma_UU, self.oma_t_out, 
                    complex_mode_shape=True, MAC_Lim=0.85, plot_type='none', show_mode_shape_number=False
                )
                time_start, time_stop = [0]*len(target_freqs), [0]*len(target_freqs)
                EFDD_damps, EFDD_Natural_frequencies = FDD_Damping(
                    SDOF, t_sdof, [time_start, time_stop], 'all_peaks', PSD_method='svd', show_mode_shape_number=False
                )
                plt.close('all') 
                
                n_modos = len(EFDD_Natural_frequencies)
                for i in range(n_modos):
                    f_val = EFDD_Natural_frequencies[i].item() if hasattr(EFDD_Natural_frequencies[i], 'item') else EFDD_Natural_frequencies[i]
                    z_val = EFDD_damps[i].item() if hasattr(EFDD_damps[i], 'item') else EFDD_damps[i]
                    modo = EFDD_Mode_shapes[i] if isinstance(EFDD_Mode_shapes, list) else EFDD_Mode_shapes[:, i]
                    
                    self.oma_results['freqs'].append(f_val)
                    self.oma_results['zetas'].append(z_val)
                    self.oma_results['modes'].append(modo)

            elif "SSI" in method:
                SSI_Coord_dict, SSI_Stable_freqs, SSI_Stable_zetas, SSI_Stable_phis = SSI_Alg(
                    y=self.oma_dados_efdd, order=200, lag=100, dt=self.oma_dt_novo, 
                    criteria=[0.01, 0.05, 0.02], Num_poles_to_accept_a_mode=5
                )
                plt.close('all') 
                
                freqs_array = np.array([float(f) for f in SSI_Stable_freqs])
                
                for target in target_freqs:
                    if len(freqs_array) == 0: continue
                    idx = np.argmin(np.abs(freqs_array - target))
                    f_val = freqs_array[idx]
                    z_val = float(SSI_Stable_zetas[idx])
                    
                    phis = SSI_Stable_phis
                    item = phis[idx] if isinstance(phis, list) else phis[:, idx]
                    if isinstance(item, torch.Tensor): item = item.detach().cpu().numpy().flatten()
                    else: item = np.array(item).flatten()
                    
                    self.oma_results['freqs'].append(f_val)
                    self.oma_results['zetas'].append(z_val)
                    self.oma_results['modes'].append(item)

            self.c_oma_view_mode.blockSignals(True)
            self.c_oma_view_mode.clear()
            
            while len(self.ax_oma_svd.lines) > 2:
                self.ax_oma_svd.lines[-1].remove()

            for i in range(len(self.oma_results['freqs'])):
                f = self.oma_results['freqs'][i]
                z = self.oma_results['zetas'][i] * 100.0
                
                row = self.table_oma_res.rowCount()
                self.table_oma_res.insertRow(row)
                self.table_oma_res.setItem(row, 0, QTableWidgetItem(str(i+1)))
                self.table_oma_res.setItem(row, 1, QTableWidgetItem(f"{f:.4f}"))
                self.table_oma_res.setItem(row, 2, QTableWidgetItem(f"{z:.4f}"))
                self.c_oma_view_mode.addItem(f"Modo {i+1}: {f:.2f} Hz, amort: {z:.2f}%")
                
                self.ax_oma_svd.axvline(x=f, color='green', linestyle='--', alpha=0.7)
                
            self.c_oma_view_mode.blockSignals(False)
            self.canvas_oma_sig.draw()
            
            self.btn_save_res_oma.setEnabled(True)
            self.plot_3d_mode()
            self.tabs_oma_plots.setCurrentIndex(1) 
            QMessageBox.information(self, "Sucesso", "Parâmetros OMA extraídos com sucesso!")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro OMA", f"Falha na extração modal:\n{str(e)}")

    def plot_3d_mode(self):
        if self.current_mode == 'oma':
            if not hasattr(self, 'c_oma_view_mode'): return
            mode_idx = self.c_oma_view_mode.currentIndex()
            if mode_idx < 0 or not hasattr(self, 'oma_results') or not self.oma_results['modes']: return
            
            nodes, lines = self.get_geometry_data()
            self.fig_oma_3d.clf()
            ax = self.fig_oma_3d.add_subplot(111, projection='3d')
            
            A_shapes = np.array(self.oma_results['modes']).T 
            dofs = self.oma_dofs_parsed
            title = self.c_oma_view_mode.currentText()
            forcar_eixo = self.c_oma_eixo.currentIndex()
            usar_ref = False 
            
            canvas = self.canvas_oma_3d
            fig = self.fig_oma_3d
            escala_manual = self.s_oma_escala.value()
            
        elif self.current_mode == 'ema':
            if not hasattr(self, 'c_view_mode'): return
            mode_idx = self.c_view_mode.currentIndex()
            if mode_idx < 0 or not hasattr(self, 'A_shapes'): return
            
            nodes, lines = self.get_geometry_data()
            self.fig_ema.clf()
            ax = self.fig_ema.add_subplot(111, projection='3d')
            
            A_shapes = self.A_shapes
            dofs = self.frf_dofs
            title = self.c_view_mode.currentText()
            forcar_eixo = self.c_eixo.currentIndex()
            
            map_text = self.c_map_base.currentText()
            usar_ref = False
            if "Auto" in map_text:
                refs = [d['ref_node'] for d in self.frf_dofs]
                rsps = [d['rsp_node'] for d in self.frf_dofs]
                if len(set(refs)) > 1 and len(set(rsps)) == 1:
                    usar_ref = True
                else:
                    usar_ref = False
            elif "Referência" in map_text:
                usar_ref = True
                
            canvas = self.canvas_ema
            fig = self.fig_ema
            escala_manual = self.s_escala.value()
        else:
            return

        for n1, n2 in lines:
            if n1 in nodes and n2 in nodes:
                p1, p2 = nodes[n1], nodes[n2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', linestyle='--', alpha=0.3, lw=1.5)
                
        node_disp = {nid: np.zeros(3) for nid in nodes}
        
        for i, dof in enumerate(dofs):
            if i >= A_shapes.shape[0]: break
            nid = dof['ref_node'] if usar_ref and 'ref_node' in dof else dof['rsp_node']
            direction = dof['ref_dir'] if usar_ref and 'ref_dir' in dof else dof['rsp_dir']
            
            amp = A_shapes[i, mode_idx]
            ref_idx = np.argmax(np.abs(A_shapes[:, mode_idx]))
            real_amp = np.real(amp * np.exp(-1j * np.angle(A_shapes[ref_idx, mode_idx])))
            
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
        scale = (0.20 * max_geom / max_d) * escala_manual if max_d > 0 else 1.0 
        
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
                
        ax.set_title(title, pad=15, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Y (mm)', fontweight='bold')
        ax.set_zlabel('Z (mm)', fontweight='bold')
        
        if nodes:
            all_x, all_y, all_z = [], [], []
            for nid in nodes:
                undef = np.array(nodes[nid])
                p_def = undef + scale * node_disp[nid]
                all_x.extend([undef[0], p_def[0]])
                all_y.extend([undef[1], p_def[1]])
                all_z.extend([undef[2], p_def[2]])
            dx = max(max(all_x) - min(all_x), 1.0); dy = max(max(all_y) - min(all_y), 1.0); dz = max(max(all_z) - min(all_z), 1.0)
            ax.set_xlim(min(all_x) - dx*0.1, max(all_x) + dx*0.1)
            ax.set_ylim(min(all_y) - dy*0.1, max(all_y) + dy*0.1)
            ax.set_zlim(min(all_z) - dz*0.1, max(all_z) + dz*0.1)
            try: ax.set_box_aspect((dx, max(dy, max(dx,dy,dz)*0.3), max(dz, max(dx,dy,dz)*0.3)))
            except AttributeError: pass
            
        ax.view_init(elev=20, azim=-60)
        fig.tight_layout()
        canvas.draw()

if __name__ == '__main__':
    if not HAS_PYQT:
        print("Erro: PyQt5 não instalado.")
        sys.exit(1)
        
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    visualizer = MAVisualizer()
    visualizer.show()
    sys.exit(app.exec_())