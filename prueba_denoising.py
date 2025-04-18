# adaptive_denoising_test.py
import numpy as np
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal


class WaveletDenoising:
    """Clase básica para denoising con wavelets (simplificada)."""
    
    def __init__(self, normalize=True, wavelet='db4', level=3, thr_mode='soft', 
                 method="universal"):
        self.normalize = normalize
        self.wavelet = wavelet
        self.level = level
        self.thr_mode = thr_mode
        self.method = method
    
    def fit(self, data):
        """Aplica el denoising a los datos de entrada."""
        # Convertir a array numpy
        data = np.asarray(data)
        
        # Normalizar si es necesario
        if self.normalize:
            data = (data - np.mean(data)) / (np.std(data) if np.std(data) != 0 else 1.0)
        
        # Descomponer con wavelet
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # Umbral
        if self.method == "universal":
            sigma = self._estimate_sigma(coeffs[-1])
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
        else:
            threshold = self._estimate_level_dependent_threshold(coeffs)
        
        # Aplicar umbral
        new_coeffs = []
        new_coeffs.append(coeffs[0])  # Aproximación
        for i in range(1, len(coeffs)):
            if self.thr_mode == 'soft':
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            else:
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
        
        # Reconstruir
        return pywt.waverec(new_coeffs, self.wavelet)
    
    def _estimate_sigma(self, detail_coeffs):
        """Estima la desviación estándar del ruido."""
        return np.median(np.abs(detail_coeffs)) / 0.6745
    
    def _estimate_level_dependent_threshold(self, coeffs):
        """Estima umbrales dependientes del nivel."""
        # Implementación simple para pruebas
        return np.std(coeffs[-1]) * 3.0


class OptimalWaveletSelector:
    """
    Implementación para selección automática de wavelets óptimas.
    """
    
    def __init__(self, max_level=None, sparsity_change_threshold=0.05):
        self.max_level = max_level
        self.sparsity_change_threshold = sparsity_change_threshold
        self.optimal_wavelet = None
        self.optimal_level = None
        self.wavelet_space = self._create_wavelet_space()
    
    def _create_wavelet_space(self):
        """Crea el espacio de búsqueda de wavelets."""
        # Lista simplificada para pruebas
        wavelet_families = [
            'db2', 'db3', 'db4','db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11',
            'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7',
            'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6',
            'coif1', 'coif2', 'coif3', 'coif4', 'coif5',
            'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8'
        ]
        return wavelet_families
    
    def calculate_sparsity(self, coeffs):
        """Calcula la esparsidad para cada nivel de coeficientes."""
        sparsity = []
        for coeff in coeffs:
            if len(coeff) > 0:
                max_abs = np.max(np.abs(coeff))
                sum_abs = np.sum(np.abs(coeff))
                if sum_abs > 0:
                    sparsity.append(max_abs / sum_abs)
                else:
                    sparsity.append(0)
            else:
                sparsity.append(0)
        return np.array(sparsity)
    
    def calculate_sparsity_change(self, sparsity):
        """Calcula el cambio de esparsidad entre niveles consecutivos."""
        sparsity_change = np.zeros_like(sparsity)
        for i in range(1, len(sparsity)):
            sparsity_change[i] = sparsity[i] - sparsity[i-1]
        return sparsity_change
    
    def find_optimal_level(self, sparsity_change):
        """Encuentra el nivel óptimo basado en el cambio de esparsidad."""
        optimal_level = 0
        for i in range(1, len(sparsity_change)):
            if sparsity_change[i] > self.sparsity_change_threshold:
                optimal_level = i - 1
                break
        
        # Si no se encontró nivel óptimo, usar el penúltimo nivel
        if optimal_level == 0 and len(sparsity_change) > 1:
            optimal_level = len(sparsity_change) - 2
        
        return optimal_level
    
    def calculate_mean_sparsity_change(self, sparsity, optimal_level):
        """Calcula el cambio medio de esparsidad."""
        if optimal_level < 1:
            return sparsity[0] if len(sparsity) > 0 else 0
        else:
            return (sparsity[optimal_level+1] - sparsity[0]) / optimal_level
    
    def find_optimal_wavelet(self, signal, n_best=1):
        """
        Encuentra la wavelet óptima para una señal.
        
        Args:
            signal: Señal de entrada.
            n_best: Número de mejores wavelets a devolver.
            
        Returns:
            Lista de tuplas (wavelet, nivel, μsc) ordenadas por μsc.
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
                # Obtener objeto wavelet
                print(f"\nProbando wavelet: {wavelet_name}")
                wavelet_obj = pywt.Wavelet(wavelet_name)
                
                # Punto crítico: Aquí es probable que ocurra el error
                print(f"  Obteniendo filtros: dec_lo = {wavelet_obj.dec_lo}")
                filter_len = len(wavelet_obj.dec_lo)
                
                # Calcular nivel máximo para esta wavelet
                max_possible_level = pywt.dwt_max_level(signal_length, filter_len)
                max_level = min(max_possible_level, self.max_level)
                print(f"  Nivel máximo para {wavelet_name}: {max_level}")
                
                # Verificar niveles suficientes
                if max_level < 1:
                    print(f"  Saltando wavelet {wavelet_name}: nivel efectivo insuficiente")
                    continue
                
                # Descomponer la señal con la wavelet actual
                print(f"  Realizando wavedec con nivel={max_level}")
                coeffs = pywt.wavedec(signal, wavelet_name, level=max_level)
                
                # Extraer coeficientes de detalle
                detail_coeffs = coeffs[1:]  # Excluir coeficientes de aproximación
                
                # Calcular esparsidad
                print(f"  Calculando esparsidad para {wavelet_name}")
                sparsity = self.calculate_sparsity(detail_coeffs)
                
                # Calcular cambio de esparsidad
                sparsity_change = self.calculate_sparsity_change(sparsity)
                
                # Encontrar nivel óptimo
                optimal_level = self.find_optimal_level(sparsity_change)
                
                # Calcular el cambio medio de esparsidad
                mean_sc = self.calculate_mean_sparsity_change(sparsity, optimal_level)
                
                print(f"  Resultados para {wavelet_name}: nivel={optimal_level}, μsc={mean_sc:.4f}")
                
                # Guardar resultados
                results.append((wavelet_name, optimal_level, mean_sc))
                
            except Exception as e:
                print(f"  Error evaluando wavelet {wavelet_name}: {str(e)}")
        
        # Verificar resultados
        if not results:
            print("ADVERTENCIA: No se pudieron encontrar wavelets óptimas.")
            # Usar wavelet predeterminada
            default_wavelet = "db4"
            default_level = 3
            default_mu_sc = 0.0
            results.append((default_wavelet, default_level, default_mu_sc))
        
        # Ordenar resultados por μsc (descendente)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Guardar la wavelet y nivel óptimos
        self.optimal_wavelet = results[0][0]
        self.optimal_level = results[0][1]
        
        print(f"\nWavelet óptima encontrada: {self.optimal_wavelet}, nivel={self.optimal_level}")
        
        # Devolver las mejores n wavelets
        return results[:n_best]


class AdaptiveWaveletDenoising:
    """
    Implementación de denoising adaptativo con wavelets.
    """
    
    def __init__(self, normalize=True, thr_mode='soft', method="universal", 
                 n_best_wavelets=5, use_optimal=True):
        self.normalize = normalize
        self.thr_mode = thr_mode
        self.method = method
        self.n_best_wavelets = n_best_wavelets
        self.use_optimal = use_optimal
        self.optimal_wavelet = None
        self.optimal_level = None
        self.best_wavelets = []
    
    def fit(self, signal):
        """
        Encuentra las mejores wavelets para la señal y aplica denoising.
        
        Args:
            signal: La señal a procesar.
            
        Returns:
            La señal procesada.
        """
        # Seleccionar wavelet óptima
        selector = OptimalWaveletSelector()
        
        try:
            self.best_wavelets = selector.find_optimal_wavelet(
                signal, n_best=self.n_best_wavelets)
            
            if len(self.best_wavelets) == 0:
                raise ValueError("No se pudieron encontrar wavelets óptimas para esta señal")
                
            self.optimal_wavelet = self.best_wavelets[0][0]
            self.optimal_level = self.best_wavelets[0][1]
            
            # Aplicar denoising con la wavelet óptima o promedio
            if self.use_optimal:
                # Usar la mejor wavelet
                denoiser = WaveletDenoising(
                    normalize=self.normalize,
                    wavelet=self.optimal_wavelet, 
                    level=self.optimal_level,
                    thr_mode=self.thr_mode,
                    method=self.method
                )
                return denoiser.fit(signal)
            else:
                # Usar promedio de las mejores wavelets
                processed_signals = []
                for wavelet, level, _ in self.best_wavelets:
                    denoiser = WaveletDenoising(
                        normalize=self.normalize,
                        wavelet=wavelet,
                        level=level, 
                        thr_mode=self.thr_mode,
                        method=self.method
                    )
                    processed_signals.append(denoiser.fit(signal))
                
                # Promediar las señales procesadas
                return np.mean(processed_signals, axis=0)
                
        except Exception as e:
            print(f"Error en AdaptiveWaveletDenoising: {str(e)}")
            # Usar parámetros predeterminados
            denoiser = WaveletDenoising(
                normalize=self.normalize,
                wavelet='db4',
                level=3,
                thr_mode=self.thr_mode, 
                method=self.method
            )
            return denoiser.fit(signal)
    
    def get_optimal_params(self):
        """Devuelve los parámetros óptimos encontrados."""
        return {
            'wavelet': self.optimal_wavelet,
            'level': self.optimal_level,
            'mu_sc': self.best_wavelets[0][2] if self.best_wavelets else 0.0,
            'all_results': self.best_wavelets
        }


def generate_test_signal(length=1024, noise_level=0.1):
    """Genera una señal de prueba con ruido."""
    t = np.linspace(0, 1, length)
    # Señal con múltiples componentes
    clean = (np.sin(2*np.pi*5*t) +   # Componente de baja frecuencia
             0.5*np.sin(2*np.pi*20*t) +  # Componente de media frecuencia
             0.25*np.sin(2*np.pi*50*t))   # Componente de alta frecuencia
    
    # Añadir ruido
    noise = noise_level * np.random.randn(length)
    noisy = clean + noise
    
    return clean, noisy, t


# Función principal de prueba
def test_adaptive_denoising():
    """Prueba la funcionalidad de denoising adaptativo."""
    print("Generando señal de prueba...")
    clean, noisy, t = generate_test_signal(length=1024, noise_level=0.2)
    
    print("\n1. Prueba con WaveletDenoising básico...")
    wd = WaveletDenoising(normalize=True, wavelet='db4', level=3)
    denoised_basic = wd.fit(noisy)
    
    print("\n2. Prueba con AdaptiveWaveletDenoising...")
    try:
        awd = AdaptiveWaveletDenoising(normalize=True, use_optimal=True, n_best_wavelets=3)
        denoised_adaptive = awd.fit(noisy)
        optimal_params = awd.get_optimal_params()
        
        print("\nParámetros óptimos encontrados:")
        print(f"Wavelet: {optimal_params['wavelet']}")
        print(f"Nivel: {optimal_params['level']}")
        print(f"μsc: {optimal_params['mu_sc']:.4f}")
        
        print("\nTop 3 wavelets:")
        for i, (wav, level, mu_sc) in enumerate(optimal_params['all_results']):
            print(f"{i+1}. {wav}, nivel={level}, μsc={mu_sc:.4f}")
        
        # Visualizar resultados
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, clean, 'g', label='Señal original')
        plt.legend()
        plt.title('Señal original limpia')
        
        plt.subplot(3, 1, 2)
        plt.plot(t, noisy, 'r', label='Señal con ruido')
        plt.legend()
        plt.title('Señal con ruido')
        
        plt.subplot(3, 1, 3)
        plt.plot(t, denoised_adaptive, 'b', label=f'Denoised ({optimal_params["wavelet"]})')
        plt.plot(t, denoised_basic, 'k--', label='Denoised (db4)', alpha=0.7)
        plt.legend()
        plt.title('Señales procesadas')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\nError durante la prueba: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_adaptive_denoising()