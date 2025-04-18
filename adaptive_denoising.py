import numpy as np
import pywt
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm
from denoising import WaveletDenoising

class OptimalWaveletSelector:
    """
    Implementación del método de selección automática de wavelets óptimas basado en
    el artículo "Optimal Wavelet Selection for Signal Denoising".
    
    Este método evalúa múltiples wavelets candidatas y selecciona la óptima 
    basada en el criterio de mean of sparsity change (μsc).
    """
    
    def __init__(self, max_level=None, sparsity_change_threshold=0.05):
        """
        Inicializa el selector de wavelets óptimas.
        
        Args:
            max_level: Nivel máximo de descomposición a considerar. Si es None,
                      se calculará automáticamente.
            sparsity_change_threshold: Umbral para determinar el nivel óptimo de 
                                     descomposición (por defecto 0.05).
        """
        self.max_level = max_level
        self.sparsity_change_threshold = sparsity_change_threshold
        self.optimal_wavelet = None
        self.optimal_level = None
        self.wavelet_space = self._create_wavelet_space()
        
    def _create_wavelet_space(self):
        """
        Crea el espacio de búsqueda de wavelets, similar a wavespace.m.
        
        Returns:
            Lista de wavelets candidatas.
        """
        # Familias de wavelets a considerar
        wave_bior = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6']
        wave_coif = ['coif1', 'coif2', 'coif3', 'coif4', 'coif5']
        wave_db = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11']
        wave_rbio = ['rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8']
        wave_sym = ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7']
        
        # Combinar todas las familias
        wave_family = wave_bior + wave_coif + wave_db + wave_rbio + wave_sym
        
        return wave_family
    
    def calculate_sparsity(self, coeffs):
        """
        Calcula la esparsidad de los coeficientes de detalle, similar a Sparsity.m.
        
        Args:
            coeffs: Lista de coeficientes de detalle.
            
        Returns:
            Vector de esparsidad para cada nivel.
        """
        sparsity = []
        for level_coeff in coeffs:
            try:
                if len(level_coeff) > 0:  # Evitar divisiones por cero
                    max_abs = np.max(np.abs(level_coeff))
                    sum_abs = np.sum(np.abs(level_coeff))
                    if sum_abs > 0:
                        sparsity.append(max_abs / sum_abs)
                    else:
                        sparsity.append(0)
                else:
                    sparsity.append(0)
            except TypeError:
                # En caso de que level_coeff no sea un array normal (como algunas wavelets biortogonales)
                print(f"  Advertencia: Tipo de coeficiente no estándar: {type(level_coeff)}")
                sparsity.append(0)  # Valor por defecto
        
        return np.array(sparsity)
    
    def calculate_sparsity_change(self, sparsity):
        """
        Calcula el cambio de esparsidad entre niveles adyacentes, similar a SparsityChange.m.
        
        Args:
            sparsity: Vector de esparsidad.
            
        Returns:
            Vector de cambio de esparsidad.
        """
        sparsity_change = np.zeros_like(sparsity)
        for i in range(1, len(sparsity)):
            sparsity_change[i] = sparsity[i] - sparsity[i-1]
        
        return sparsity_change
    
    def find_optimal_level(self, sparsity_change):
        """
        Determina el nivel de descomposición óptimo.
        
        Args:
            sparsity_change: Vector de cambio de esparsidad.
            
        Returns:
            Nivel óptimo de descomposición.
        """
        # Encontrar el primer nivel donde el cambio de esparsidad supera el umbral
        for i in range(1, len(sparsity_change)):
            if sparsity_change[i] > self.sparsity_change_threshold:
                return i - 1  # κ = j - 1
        
        # Si no se encuentra, usar el nivel máximo - 1
        return len(sparsity_change) - 2
    
    def calculate_mean_sparsity_change(self, sparsity, optimal_level):
        """
        Calcula el promedio de cambio de esparsidad, similar a Meansc.m.
        
        Args:
            sparsity: Vector de esparsidad.
            optimal_level: Nivel óptimo de descomposición.
            
        Returns:
            Promedio de cambio de esparsidad.
        """
        if optimal_level < 1:
            return sparsity[0]
        else:
            return (sparsity[optimal_level+1] - sparsity[0]) / (optimal_level)
        
    def calculate_effective_level(self, wavelet, signal_length):
        """
        Calcula el nivel efectivo de descomposición basado en la longitud del filtro.
        
        Args:
            wavelet: Nombre de la wavelet.
            signal_length: Longitud de la señal.
            
        Returns:
            Nivel efectivo de descomposición.
        """
        # Obtener la longitud del filtro de descomposición paso bajo directamente
        filter_len = len(pywt.Wavelet(wavelet).dec_lo)
        
        # Calcular el nivel máximo posible
        max_possible_level = pywt.dwt_max_level(signal_length, filter_len)
        
        # Verificar relaciones para los niveles efectivos
        effective_level = 0
        for j in range(1, max_possible_level + 1):
            detail_len = signal_length // (2**j)
            ratio = detail_len / filter_len
            if ratio <= 1.5:
                break
            effective_level = j
            
        return effective_level
        
def _is_biorthogonal(self, wavelet_name):
    """
    Verifica si una wavelet es biortogonal.
    
    Args:
        wavelet_name: Nombre de la wavelet a verificar.
        
    Returns:
        True si la wavelet es biortogonal, False en caso contrario.
    """
    return wavelet_name.startswith('bior') or wavelet_name.startswith('rbio')
        
    
def find_optimal_wavelet(self, signal, n_best=1):
    """
    Encuentra la(s) wavelet(s) óptima(s) para una señal dada.

    Args:
        signal: Señal de entrada para denoising.
        n_best: Número de mejores wavelets a devolver.
        
    Returns:
        Lista de tuplas (wavelet, nivel, μsc) ordenadas por μsc descendente.
    """
    signal = np.array(signal, dtype=np.float64)
    signal_length = len(signal)

    # Calcular el nivel máximo posible si no se proporciona
    if self.max_level is None:
        self.max_level = int(np.floor(np.log2(signal_length)))

    # Resultados para cada wavelet
    results = []

    print(f"Total de wavelets a evaluar: {len(self.wavelet_space)}")
    print(f"Longitud de la señal: {signal_length}")
    print(f"Nivel máximo considerado: {self.max_level}")

    # Evaluar cada wavelet en el espacio de búsqueda
    for wavelet_name in tqdm(self.wavelet_space, desc="Evaluando wavelets"):
        try:
            wavelet_obj = pywt.Wavelet(wavelet_name)
            
            # Obtener la longitud del filtro de descomposición paso bajo
            filter_len = len(wavelet_obj.dec_lo)
            
            # Calcular el nivel máximo posible para esta wavelet
            max_possible_level = pywt.dwt_max_level(signal_length, filter_len)
            max_level = min(max_possible_level, self.max_level)
            
            # Verificar que tengamos suficientes niveles para descomponer
            if max_level < 1:
                print(f"Saltando wavelet {wavelet_name}: nivel efectivo insuficiente")
                continue
            
            # Descomponer la señal usando la wavelet actual
            coeffs = pywt.wavedec(signal, wavelet_name, level=max_level)
            
            # Extraer coeficientes de detalle (excluir aproximación)
            detail_coeffs = coeffs[1:]
            
            # Calcular esparsidad para cada nivel - usar el método de la clase
            sparsity = self.calculate_sparsity(detail_coeffs)
            
            # Calcular cambio de esparsidad
            sparsity_change = self.calculate_sparsity_change(sparsity)
            
            # Encontrar nivel óptimo de descomposición
            optimal_level = self.find_optimal_level(sparsity_change)
            
            # Calcular el mean sparsity change (μsc)
            mean_sc = self.calculate_mean_sparsity_change(sparsity, optimal_level)
            
            # Guardar los resultados
            results.append((wavelet_name, optimal_level, mean_sc))
            
        except Exception as e:
            print(f"Error evaluando wavelet {wavelet_name}: {str(e)}")

    # Verificar si tenemos resultados
    if not results:
        print("ADVERTENCIA: No se pudo encontrar ninguna wavelet óptima.")
        # En lugar de fallar, usar una wavelet predeterminada con configuraciones seguras
        default_wavelet = "db4" 
        default_level = 3
        default_mu_sc = 0.0
        results.append((default_wavelet, default_level, default_mu_sc))
        print(f"Usando wavelet predeterminada: {default_wavelet}, nivel={default_level}")

    # Ordenar los resultados por μsc (descendente)
    results.sort(key=lambda x: x[2], reverse=True)

    # Guardar la wavelet y nivel óptimos (el primero de la lista)
    self.optimal_wavelet = results[0][0]
    self.optimal_level = results[0][1]

    # Devolver las mejores n_best wavelets
    return results[:n_best]

class AdaptiveWaveletDenoising:
    """
    Clase mejorada de WaveletDenoising que selecciona automáticamente
    la wavelet y nivel óptimos para el denoising de señales.
    """
    
    def __init__(self, normalize=True, thr_mode='soft', method="universal", 
                 n_best_wavelets=5, use_optimal=True, recon_mode='smooth'):
        """
        Constructor de AdaptiveWaveletDenoising.
        
        Args:
            normalize: Habilita la normalización de la señal de entrada en [0, 1]
            thr_mode: Tipo de umbralización ('soft' o 'hard')
            method: Tipo de método para determinar el umbral
            n_best_wavelets: Número de mejores wavelets a considerar
            use_optimal: Si es True, usa la mejor wavelet; si es False, evalúa todas y promedia
            recon_mode: Modo de extensión de la señal para la reconstrucción
        """
        self.normalize = normalize
        self.thr_mode = thr_mode
        self.method = method
        self.recon_mode = recon_mode
        self.n_best_wavelets = n_best_wavelets
        self.use_optimal = use_optimal
        self.optimal_selector = OptimalWaveletSelector()
        self.best_wavelets = None
        self.scaler = None
        self.normalized_data = None
        
    def fit(self, signal):
        """
        Aplica el denoising con selección automática de wavelet.
        
        Args:
            signal: Señal de entrada a procesar
            
        Returns:
            Señal procesada con denoising
        """
        # Copiar la señal para evitar modificar la original
        signal_copy = np.array(signal, copy=True)
        
        # Preprocesar la señal (detrend y normalización)
        preprocessed_signal = self.preprocess(signal_copy)
        
        # Encontrar las mejores wavelets para esta señal
        self.best_wavelets = self.optimal_selector.find_optimal_wavelet(
            preprocessed_signal, 
            n_best=self.n_best_wavelets
        )
        
        if len(self.best_wavelets) == 0:
            raise ValueError("No se pudieron encontrar wavelets óptimas para esta señal")
        
        # Mostrar las mejores wavelets encontradas
        print("Mejores wavelets encontradas:")
        for i, (wavelet, level, mu_sc) in enumerate(self.best_wavelets):
            print(f"{i+1}. {wavelet}, nivel={level}, μsc={mu_sc:.4f}")
        
        # Aplicar denoising según la estrategia seleccionada
        if self.use_optimal:
            # Usar solo la mejor wavelet
            best_wavelet, best_level, _ = self.best_wavelets[0]
            print(f"\nUsando la mejor wavelet: {best_wavelet}, nivel={best_level}")
            
            # Crear un objeto WaveletDenoising con los parámetros óptimos
            wd = WaveletDenoising(
                normalize=False,  # Ya normalizamos antes
                wavelet=best_wavelet,
                level=best_level,
                thr_mode=self.thr_mode,
                recon_mode=self.recon_mode,
                method=self.method
            )
            
            # Aplicar denoising y guardar la señal resultante
            denoised_signal = wd.fit(preprocessed_signal)
            
        else:
            # Promediar los resultados de las mejores n wavelets
            print(f"\nPromediando los resultados de las {len(self.best_wavelets)} mejores wavelets")
            denoised_signals = []
            
            for wavelet, level, _ in self.best_wavelets:
                wd = WaveletDenoising(
                    normalize=False,
                    wavelet=wavelet,
                    level=level,
                    thr_mode=self.thr_mode,
                    recon_mode=self.recon_mode,
                    method=self.method
                )
                denoised = wd.fit(preprocessed_signal)
                denoised_signals.append(denoised)
            
            # Promediar todas las señales procesadas
            denoised_signal = np.mean(denoised_signals, axis=0)
        
        # Desnormalizar si es necesario
        if self.normalize:
            denoised_signal = self.scaler.inverse_transform(
                denoised_signal.reshape(-1, 1))[:, 0]
        
        return denoised_signal
        
    def preprocess(self, signal):
        """
        Preprocesa la señal: elimina tendencias y normaliza si es necesario.
        
        Args:
            signal: Señal de entrada
            
        Returns:
            Señal preprocesada
        """
        # Eliminar tendencias (DC, etc.)
        signal_detrended = detrend(signal, type='constant')
        
        # Normalizar los datos a [0, 1] y guardar el escalador para uso futuro
        if self.normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            signal_normalized = self.scaler.fit_transform(signal_detrended.reshape(-1, 1))[:, 0]
            self.normalized_data = signal_normalized.copy()
            return signal_normalized
        
        return signal_detrended
    
    def get_optimal_params(self):
        """
        Devuelve los parámetros óptimos seleccionados.
        
        Returns:
            Diccionario con la wavelet y nivel óptimos
        """
        if self.best_wavelets:
            best_wavelet, best_level, mu_sc = self.best_wavelets[0]
            return {
                'wavelet': best_wavelet,
                'level': best_level,
                'mu_sc': mu_sc,
                'all_results': self.best_wavelets
            }
        else:
            return None