"""
smartlight_puf_v2.py
====================
Librería completa del simulador SmartLight PUF.
Código extraído directamente del notebook 10_03_2026_cuantizarfases.

Contiene:
    - TBU_SNM                              : matriz de transferencia de una TBU
    - malla_hexagonal                      : geometría + cables
    - malla_SNM                            : grafo de flujo de señal
    - malla_SNM_completa                   : ADN PUF + programación
    - generar_configuracion_totalmente_aleatoria
    - procesar_un_puerto
    - ejecutar_y_guardar_chip
    - ejecutar_simulacion_masiva           : bucle paralelo (joblib)
    - binarizar_puf_lehmer_por_columnas    : Lehmer + Gray
    - binarizar_puf_rapido                 : versión vectorizada
    - fast_hamming_sampling                : HD por muestreo
    - generar_llaves_npz                   : guarda llaves_binarizadas.npz
    - analizar_seguridad_puf               : análisis básico
    - analizar_puf_tfm_final               : análisis completo pre/post hash
    - rutas_maestras_anillo                : dict de rutas (40 puertos)
    - rutas_maestras                       : dict de rutas directas

Uso básico:
    from smartlight_puf_v2 import (
        malla_SNM_completa,
        ejecutar_simulacion_masiva,
        generar_llaves_npz,
        analizar_puf_tfm_final,
        rutas_maestras_anillo,
        rutas_maestras,
    )
    ejecutar_simulacion_masiva(n_chips=100, n_retos=10,
                               rutas_anillo=rutas_maestras_anillo,
                               directorio='mi_db')
    generar_llaves_npz('mi_db')
    analizar_puf_tfm_final('llaves_binarizadas_lehmer_columnas.npz')
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import os
import math
import random
import hashlib

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(x, **kw): return x

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT = True
except ImportError:
    _PLOT = False


# ===========================================================================
# 1. MODELO FÍSICO DE LA TBU
# ===========================================================================

def TBU_SNM(p, phi1, phi2, omega=0, tau=0):
    """
    TBU bidireccional usando las ecuaciones de Capmany mapeadas a SNM.
    Incluye sensibilidad a la frecuencia (omega) y retardo (tau).
    Código idéntico al notebook original.
    """
    c_a, s_a = np.sqrt(1 - p['K_a']), np.sqrt(p['K_a'])
    c_b, s_b = np.sqrt(1 - p['K_b']), np.sqrt(p['K_b'])

    multiplicador = (np.sqrt(1 - p['gamma_a']) *
                     np.sqrt(1 - p['gamma_b']) *
                     np.exp(-1j * omega * tau))

    theta1 = phi1 + omega * tau
    theta2 = phi2 + omega * tau

    m11 = (c_a*c_b*np.sqrt(1-p['gamma_c'])*np.exp(-1j*theta1)
           - s_a*s_b*np.sqrt(1-p['gamma_d'])*np.exp(-1j*theta2))
    m12 = (-1j*s_a*c_b*np.sqrt(1-p['gamma_c'])*np.exp(-1j*theta1)
           - 1j*s_b*c_a*np.sqrt(1-p['gamma_d'])*np.exp(-1j*theta2))
    m21 = (-1j*s_b*c_a*np.sqrt(1-p['gamma_c'])*np.exp(-1j*theta1)
           - 1j*s_a*c_b*np.sqrt(1-p['gamma_d'])*np.exp(-1j*theta2))
    m22 = (c_a*c_b*np.sqrt(1-p['gamma_d'])*np.exp(-1j*theta2)
           - s_a*s_b*np.sqrt(1-p['gamma_c'])*np.exp(-1j*theta1))

    S_local = np.zeros((4, 4), dtype=complex)
    S_local[2, 0] = m11;  S_local[2, 1] = m12
    S_local[3, 0] = m21;  S_local[3, 1] = m22
    S_local[0, 2] = m11;  S_local[1, 2] = m12
    S_local[0, 3] = m21;  S_local[1, 3] = m22

    return multiplicador * S_local


# ===========================================================================
# 2. MALLA HEXAGONAL (capa base)
# ===========================================================================

class malla_hexagonal:
    def __init__(self, tbus_por_columna, ideal=False, seed=None):
        self.tbus_reales = tbus_por_columna
        self.Cols  = len(tbus_por_columna)
        self.Filas = max(tbus_por_columna)
        self.ideal = ideal
        self.num_puertos = self.Cols * self.Filas * 4
        self.S = sp.lil_matrix((self.num_puertos, self.num_puertos), dtype=complex)
        self.seed = seed
        self.params_tbus = {}
        self.cable_loss = 1.0 if ideal else 1
        self._inicializar_defectos()

    def _existe_tbu(self, c, f):
        if c < 0 or c >= self.Cols: return False
        if f < 0 or f >= self.tbus_reales[c]: return False
        return True

    def _get_ids(self, c, f):
        base = (c * self.Filas + f) * 4
        return base, base+1, base+2, base+3

    def _inicializar_defectos(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        K_mean, K_std = 0.5, 0.0433
        g_c_m, g_c_s  = 0.0115, 0.01
        g_a_m, g_a_s  = 0.0516, 0.0284
        for c in range(self.Cols):
            for f in range(self.tbus_reales[c]):
                if self.ideal:
                    self.params_tbus[(c, f)] = {
                        'K_a': 0.5, 'K_b': 0.5,
                        'gamma_a': 0.0, 'gamma_b': 0.0,
                        'gamma_c': 0.0, 'gamma_d': 0.0
                    }
                else:
                    self.params_tbus[(c, f)] = {
                        'K_a':     np.clip(np.random.normal(K_mean, K_std), 0.0, 1.0),
                        'K_b':     np.clip(np.random.normal(K_mean, K_std), 0.0, 1.0),
                        'gamma_a': np.clip(np.random.normal(g_c_m,  g_c_s),  0.0, 1.0),
                        'gamma_b': np.clip(np.random.normal(g_c_m,  g_c_s),  0.0, 1.0),
                        'gamma_c': np.clip(np.random.normal(g_a_m,  g_a_s),  0.0, 1.0),
                        'gamma_d': np.clip(np.random.normal(g_a_m,  g_a_s),  0.0, 1.0),
                    }

    def _conectar_cables(self):
        cable = self.cable_loss
        for c in range(self.Cols):
            tipo  = c % 4
            num_f = self.tbus_reales[c]
            for f in range(num_f):
                it, ib, ot, ob = self._get_ids(c, f)

                # A. Conexiones verticales
                if (tipo == 0 or tipo == 2) and f < num_f - 1:
                    d_it, d_ib, d_ot, d_ob = self._get_ids(c, f + 1)
                    if tipo == 0:
                        if f % 2 == 0:
                            self.S[ob, d_ot] = cable;  self.S[d_ot, ob] = cable
                        else:
                            self.S[ib, d_it] = cable;  self.S[d_it, ib] = cable
                    elif tipo == 2:
                        if f % 2 == 0:
                            self.S[ib, d_it] = cable;  self.S[d_it, ib] = cable
                        else:
                            self.S[ob, d_ot] = cable;  self.S[d_ot, ob] = cable

                # B. Conexiones horizontales
                if c < self.Cols - 1:
                    if tipo == 0:  # Contracción 6->3
                        f_dest = f // 2
                        target = self._get_ids(c + 1, f_dest)
                        if f % 2 == 1:
                            self.S[ob, target[1]] = cable;  self.S[target[1], ob] = cable
                        else:
                            self.S[ot, target[0]] = cable;  self.S[target[0], ot] = cable

                    elif tipo == 1:  # Expansión 3->6
                        t_sup = self._get_ids(c + 1, 2 * f)
                        self.S[ot, t_sup[0]] = cable;  self.S[t_sup[0], ot] = cable
                        t_inf = self._get_ids(c + 1, 2 * f + 1)
                        self.S[ob, t_inf[1]] = cable;  self.S[t_inf[1], ob] = cable

                    elif tipo == 2:  # Contracción 6->4
                        if f == 0:
                            target = self._get_ids(c + 1, 0)
                            self.S[ob, target[1]] = cable;  self.S[target[1], ob] = cable
                        elif f == num_f - 1:
                            target = self._get_ids(c + 1, self.tbus_reales[c+1] - 1)
                            self.S[ot, target[0]] = cable;  self.S[target[0], ot] = cable
                        else:
                            f_dest = ((f - 1) // 2) + 1
                            target = self._get_ids(c + 1, f_dest)
                            if f % 2 == 0:
                                self.S[ob, target[1]] = cable;  self.S[target[1], ob] = cable
                            else:
                                self.S[ot, target[0]] = cable;  self.S[target[0], ot] = cable

                    elif tipo == 3:  # Expansión 4->6
                        if f == 0:
                            target = self._get_ids(c + 1, 0)
                            self.S[ob, target[1]] = cable;  self.S[target[1], ob] = cable
                        elif f == num_f - 1:
                            target = self._get_ids(c + 1, self.tbus_reales[c+1] - 1)
                            self.S[ot, target[0]] = cable;  self.S[target[0], ot] = cable
                        else:
                            t_v_sup = self._get_ids(c + 1, 2 * f - 1)
                            self.S[ot, t_v_sup[0]] = cable;  self.S[t_v_sup[0], ot] = cable
                            t_v_inf = self._get_ids(c + 1, 2 * f)
                            self.S[ob, t_v_inf[1]] = cable;  self.S[t_v_inf[1], ob] = cable

    def detectar_puertos_externos(self):
        if self.S.nnz == 0:
            self._conectar_cables()
        S_csr = self.S.tocsr()
        puertos_externos = []
        for c in range(self.Cols):
            for f in range(self.tbus_reales[c]):
                ids_tbu = self._get_ids(c, f)
                for p_id in ids_tbu:
                    if S_csr[p_id, :].nnz == 0:
                        puertos_externos.append(p_id)
        return puertos_externos

    def obtener_ids_perimetrales_ordenados(self):
        ids_libres = self.detectar_puertos_externos()
        info_puertos = []
        for p_id in ids_libres:
            total_tbu_id = p_id // 4
            c       = total_tbu_id // self.Filas
            f       = total_tbu_id % self.Filas
            p_local = p_id % 4
            x = c
            y = f * 2 + (1 if c % 2 != 0 else 0)
            if c == 0:               borde = 'WEST'
            elif c == self.Cols - 1: borde = 'EAST'
            elif p_local in (0, 2):  borde = 'NORTH'
            else:                    borde = 'SOUTH'
            info_puertos.append({'id': p_id, 'c': c, 'f': f,
                                  'p_local': p_local, 'x': x, 'y': y, 'borde': borde})

        inicio      = sorted([p for p in info_puertos if p['c'] == 0 and p['f'] == 0],
                              key=lambda x: x['p_local'], reverse=True)
        norte       = sorted([p for p in info_puertos if p['borde'] == 'NORTH' and 0 < p['c'] < self.Cols-1],
                              key=lambda x: x['x'])
        este        = sorted([p for p in info_puertos if p['borde'] == 'EAST'],
                              key=lambda x: x['y'])
        sur         = sorted([p for p in info_puertos if p['borde'] == 'SOUTH' and 0 < p['c'] < self.Cols-1],
                              key=lambda x: x['x'], reverse=True)
        oeste_resto = sorted([p for p in info_puertos if p['c'] == 0 and p['f'] > 0],
                              key=lambda x: (x['y'], x['id']), reverse=True)

        return [p['id'] for p in inicio + norte + este + sur + oeste_resto]


# ===========================================================================
# 3. MALLA SNM (grafo de flujo de señal)
# ===========================================================================

class malla_SNM(malla_hexagonal):

    def __init__(self, tbus_por_columna, ideal=False, seed=None):
        super().__init__(tbus_por_columna, ideal, seed)
        self.num_nodos = self.num_puertos * 2
        self.G = sp.lil_matrix((self.num_nodos, self.num_nodos), dtype=complex)

    def _get_nodos(self, p_id):
        """Retorna los índices (nodo_entrada, nodo_salida) para un puerto global."""
        return p_id * 2, p_id * 2 + 1

    def construir_grafo(self, fases_input):
        """Mapea la estructura al grafo de flujo inicial."""
        self.G = sp.lil_matrix((self.num_nodos, self.num_nodos), dtype=complex)

        # 1. Conexiones internas (TBUs)
        for (c, f), p in self.params_tbus.items():
            phi1, phi2 = fases_input.get((c, f), (0, 0))
            S_tbu = TBU_SNM(p, phi1, phi2)
            ids_puertos = self._get_ids(c, f)
            for i_local in range(4):
                for j_local in range(4):
                    if S_tbu[j_local, i_local] != 0:
                        n_in  = self._get_nodos(ids_puertos[i_local])[0]
                        n_out = self._get_nodos(ids_puertos[j_local])[1]
                        self.G[n_out, n_in] = S_tbu[j_local, i_local]

        # 2. Conexiones de cables
        malla_temp = malla_hexagonal(self.tbus_reales, self.ideal)
        malla_temp._conectar_cables()
        conn = malla_temp.S.tocoo()
        for p1, p2 in zip(conn.row, conn.col):
            if p1 < p2:
                n1_in, n1_out = self._get_nodos(p1)
                n2_in, n2_out = self._get_nodos(p2)
                self.G[n2_in, n1_out] = self.cable_loss
                self.G[n1_in, n2_out] = self.cable_loss

    def resolver_matriz_total(self, ids_externos):
        """Calcula la matriz de scattering total entre los puertos externos."""
        I     = sp.eye(self.num_nodos, format='csr')
        A     = (I - self.G.tocsr()).tocsc()
        solve = factorized(A)

        num_ext = len(ids_externos)
        S_total = np.zeros((num_ext, num_ext), dtype=complex)
        for j in range(num_ext):
            b = np.zeros(self.num_nodos, dtype=complex)
            n_in_j = self._get_nodos(ids_externos[j])[0]
            b[n_in_j] = 1.0
            x = solve(b)
            for i in range(num_ext):
                n_out_i = self._get_nodos(ids_externos[i])[1]
                S_total[i, j] = x[n_out_i]
        return S_total


# ===========================================================================
# 4. MALLA SNM COMPLETA (ADN PUF + programación)
# ===========================================================================

class malla_SNM_completa(malla_SNM):

    def __init__(self, tbus_por_columna, ideal=False, seed=None, bits_res_fase=8):
        super().__init__(tbus_por_columna, ideal, seed)
        self.mapa_tbus      = self._generar_mapa_tbus()
        self.bits_res_fase  = bits_res_fase

        # El seed ya fue aplicado en _inicializar_defectos (malla_hexagonal).
        # Las fases pasivas continúan la misma secuencia aleatoria del notebook.
        if seed is not None:
            np.random.seed(seed)

        self.fases_pasivas = {}
        for coords in self.params_tbus.keys():
            f1_raw = np.random.uniform(0, 2*np.pi)
            f2_raw = np.random.uniform(0, 2*np.pi)
            self.fases_pasivas[coords] = (
                self._cuantizar(f1_raw),
                self._cuantizar(f2_raw)
            )

    def _cuantizar(self, fase):
        """Cuantiza una fase a la resolución del DAC interno."""
        if self.bits_res_fase is None:
            return fase
        niveles = 2**self.bits_res_fase
        return np.round(fase * niveles / (2 * np.pi)) * (2 * np.pi) / niveles

    def _generar_mapa_tbus(self):
        """Crea el diccionario: Índice Lineal -> (Columna, Fila)"""
        mapa, contador = {}, 0
        for c in range(self.Cols):
            for f in range(self.tbus_reales[c]):
                mapa[contador] = (c, f)
                contador += 1
        return mapa

    def programar_malla(self, config_reto, ids_maestros=None, omega=0, tau=0, fase_comun=0):
        """
        Lógica de Programación:
        - TBU 'Maestra' (ids_maestros): Se calibra (borra el ADN).
        - Resto: ADN + Ruido del reto.
        """
        ids_maestros  = set(ids_maestros) if ids_maestros else set()
        fases_finales = {}
        dict_reto     = dict(config_reto)

        for idx_lineal, coords in self.mapa_tbus.items():
            f_pasiva = self.fases_pasivas[coords]
            f_reto   = dict_reto.get(idx_lineal, (0, 0))

            if f_reto == "x":
                phi_r = (0, 0)
            elif f_reto == "=":
                phi_r = (np.pi, 0)
            elif isinstance(f_reto, (int, float)):
                phi_r = (2 * np.arcsin(np.sqrt(f_reto)), 0)
            else:
                phi_r = f_reto

            if idx_lineal in ids_maestros:
                fases_finales[coords] = phi_r
            else:
                fases_finales[coords] = (
                    f_pasiva[0] + phi_r[0] + fase_comun,
                    f_pasiva[1] + phi_r[1] + fase_comun,
                )

        self.construir_grafo_espectral(fases_finales, omega, tau)

    def construir_grafo_espectral(self, fases_input, omega, tau):
        """Construye el grafo de flujo de señal con dependencia espectral."""
        self.G = sp.lil_matrix((self.num_nodos, self.num_nodos), dtype=complex)

        # 1. Conexiones internas de las TBUs con dependencia espectral
        for (c, f), p in self.params_tbus.items():
            phi1, phi2 = fases_input.get((c, f), (0, 0))
            S_tbu = TBU_SNM(p, phi1, phi2, omega, tau)
            ids_puertos = self._get_ids(c, f)
            for i_local in range(4):
                for j_local in range(4):
                    if S_tbu[j_local, i_local] != 0:
                        n_in  = self._get_nodos(ids_puertos[i_local])[0]
                        n_out = self._get_nodos(ids_puertos[j_local])[1]
                        self.G[n_out, n_in] = S_tbu[j_local, i_local]

        # 2. Conexiones de cables con fase de retardo
        fase_cable = np.exp(-1j * omega * tau)
        malla_temp = malla_hexagonal(self.tbus_reales, self.ideal)
        malla_temp._conectar_cables()
        conn = malla_temp.S.tocoo()
        for p1, p2 in zip(conn.row, conn.col):
            if p1 < p2:
                n1_in, n1_out = self._get_nodos(p1)
                n2_in, n2_out = self._get_nodos(p2)
                self.G[n2_in, n1_out] = self.cable_loss * fase_cable
                self.G[n1_in, n2_out] = self.cable_loss * fase_cable


# ===========================================================================
# 5. MOTOR DE SIMULACIÓN
# ===========================================================================

def generar_configuracion_totalmente_aleatoria(malla, seed, bits_res=8):
    """Genera retos aleatorios cuantizados para todos los índices de la malla."""
    np.random.seed(seed)
    config = []
    for idx in malla.mapa_tbus.keys():
        f1 = np.random.uniform(0, 2*np.pi)
        f2 = np.random.uniform(0, 2*np.pi)
        if bits_res:
            niveles = 2**bits_res
            f1 = np.round(f1 * niveles / (2*np.pi)) * (2*np.pi) / niveles
            f2 = np.round(f2 * niveles / (2*np.pi)) * (2*np.pi) / niveles
        config.append((idx, (f1, f2)))
    return config


def procesar_un_puerto(idx_in, p_humano, malla, ids_activos,
                        dict_rutas, ids_maestros, config_fondo):
    """
    Calcula la columna idx_in de la matriz de potencias |S|^2.
    Programa la malla con fondo + ruta específica del puerto.
    """
    fondo_dict  = dict(config_fondo)
    conf_puerto = fondo_dict.copy()
    if p_humano in dict_rutas:
        for tbu_id, estado in dict_rutas[p_humano]:
            conf_puerto[tbu_id] = estado
    malla.programar_malla(list(conf_puerto.items()), ids_maestros=ids_maestros)
    return np.abs(malla.resolver_matriz_total(ids_activos)[:, idx_in])**2


def ejecutar_y_guardar_chip(c_idx, tbus_columnas, ids_activos, indices_validos,
                              rutas_anillo, rutas_centro, res_fase, num_retos, directorio):
    """
    Simula un chip completo y guarda el resultado en .npz.
    Fichero: directorio/chip_{seed_hw}.npz
    Claves: 'base_anillo', 'anillo', 'seed_hw'
    """
    seed_hw  = 5000 + c_idx
    os.makedirs(directorio, exist_ok=True)
    filename = os.path.join(directorio, f'chip_{seed_hw}.npz')
    if os.path.exists(filename):
        return

    malla_chip = malla_SNM_completa(tbus_columnas, seed=seed_hw, bits_res_fase=res_fase)
    tbus_maestras_anillo = {t[0] for ruta in rutas_anillo.values() for t in ruta}

    N = len(ids_activos)

    # 1. Medida base
    ruido_base = generar_configuracion_totalmente_aleatoria(
        malla_chip, seed=seed_hw + 777, bits_res=res_fase)
    res_base_anillo = np.zeros((N, N), dtype=np.float32)
    for idx, p_in in enumerate(ids_activos):
        p_humano = indices_validos[idx]
        res_base_anillo[:, idx] = procesar_un_puerto(
            idx, p_humano, malla_chip, ids_activos,
            rutas_anillo, tbus_maestras_anillo, ruido_base)

    # 2. Retos
    retos_anillo = []
    for r in range(num_retos):
        ruido_reto = generar_configuracion_totalmente_aleatoria(
            malla_chip, seed=seed_hw + r + 1, bits_res=res_fase)
        m_reto = np.zeros((N, N), dtype=np.float32)
        for idx, p_in in enumerate(ids_activos):
            p_humano = indices_validos[idx]
            m_reto[:, idx] = procesar_un_puerto(
                idx, p_humano, malla_chip, ids_activos,
                rutas_anillo, tbus_maestras_anillo, ruido_reto)
        retos_anillo.append(m_reto)

    np.savez_compressed(filename,
                        base_anillo=res_base_anillo,
                        anillo=np.array(retos_anillo),
                        seed_hw=np.array(seed_hw))


def ejecutar_simulacion_masiva(n_chips=100, n_retos=10,
                                tbus_columnas=None,
                                indices_validos=None,
                                rutas_anillo=None,
                                rutas_centro=None,
                                res_fase=8,
                                directorio='base_datos_puf',
                                n_jobs=-1):
    """
    Simulación masiva de chips con paralelización via joblib (si disponible).

    Parámetros
    ----------
    n_chips       : int
    n_retos       : int
    tbus_columnas : list  (default SmartLight 15 columnas)
    indices_validos: list (default 28 puertos activos de 40)
    rutas_anillo  : dict  — rutas_maestras_anillo
    rutas_centro  : dict  — rutas_maestras
    res_fase      : int   — resolución de fase en bits
    directorio    : str
    n_jobs        : int   — núcleos para joblib (-1 = todos)
    """
    tbus_columnas   = tbus_columnas   or [6,3,6,4,6,3,6,4,6,3,6,4,6,3,6]
    indices_validos = indices_validos or [i for i in range(40) if i < 22 or i > 33]
    rutas_anillo    = rutas_anillo    or {}
    rutas_centro    = rutas_centro    or {}

    malla_ref   = malla_SNM_completa(tbus_columnas, seed=0)
    ids_40      = malla_ref.obtener_ids_perimetrales_ordenados()
    ids_activos = [ids_40[i] for i in indices_validos]

    print(f"Simulación masiva: {n_chips} chips x {n_retos} retos")
    print(f"Puertos activos  : {len(ids_activos)} | Resolución: {res_fase} bits")
    print(f"Directorio       : {directorio}/")

    if _JOBLIB and n_jobs != 1:
        Parallel(n_jobs=n_jobs)(
            delayed(ejecutar_y_guardar_chip)(
                c, tbus_columnas, ids_activos, indices_validos,
                rutas_anillo, rutas_centro, res_fase, n_retos, directorio
            ) for c in _tqdm(range(n_chips), desc='Simulando en Paralelo')
        )
    else:
        for c in _tqdm(range(n_chips), desc='Fabricando chips'):
            ejecutar_y_guardar_chip(
                c, tbus_columnas, ids_activos, indices_validos,
                rutas_anillo, rutas_centro, res_fase, n_retos, directorio)

    print(f"\n✅  Base de datos completada en '{directorio}/'")
    return directorio


# ===========================================================================
# 6. BINARIZACIÓN
# ===========================================================================

def binarizar_puf_lehmer_por_columnas(matriz_lineal):
    """
    Binariza la matriz procesando cada columna de forma independiente.
    Lehmer + código Gray. Código idéntico al notebook.
    """
    vector_bits_total = []
    for c in range(matriz_lineal.shape[1]):
        columna = 10 * np.log10(matriz_lineal[:, c] + 1e-12)
        n = len(columna)
        r = []
        for j in range(1, n):
            cuenta = np.sum(columna[:j] < columna[j])
            r.append(cuenta)
        for j, rj in enumerate(r):
            n_bits = math.ceil(math.log2(j + 2))
            gj     = rj ^ (rj >> 1)
            vector_bits_total.append(format(gj, f'0{n_bits}b'))
    return ''.join(vector_bits_total)


def binarizar_puf_rapido(matriz_lineal):
    """Versión vectorizada: devuelve array uint8 en lugar de string."""
    bits_str = binarizar_puf_lehmer_por_columnas(matriz_lineal)
    return np.array([int(b) for b in bits_str], dtype=np.uint8)


# ===========================================================================
# 7. DISTANCIAS DE HAMMING
# ===========================================================================

def calcular_hd(s1, s2):
    """Distancia de Hamming normalizada entre dos cadenas de bits."""
    diffs = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return diffs / len(s1)


def fast_hamming_sampling(bit_matrix, n_pairs=50000):
    """
    Calcula HD por muestreo aleatorio (eficiente para datasets grandes).
    bit_matrix : np.ndarray (N, n_bits) dtype uint8
    """
    n_items = bit_matrix.shape[0]
    hds = []
    for _ in range(n_pairs):
        i, j = random.sample(range(n_items), 2)
        hd = np.mean(bit_matrix[i] != bit_matrix[j])
        hds.append(float(hd))
    return hds


# ===========================================================================
# 8. PIPELINE: DB -> llaves .npz
# ===========================================================================

def generar_llaves_npz(directorio_data,
                        ruta_salida='llaves_binarizadas_lehmer_columnas.npz',
                        n_chips=None,
                        n_pairs_hd=50000):
    """
    Lee todos los .npz de chips simulados, binariza y guarda las llaves.
    Genera el fichero compatible con analizar_puf_tfm_final().

    Claves del fichero de salida:
        'llaves_unicidad'          : array de strings de bits (n_chips,)
        'llaves_independencia'     : array de strings de bits (n_retos,)
        'distancias_unicidad'      : array de HD inter-chip
        'distancias_independencia' : array de HD inter-reto
    """
    archivos = sorted([f for f in os.listdir(directorio_data) if f.endswith('.npz')])
    if n_chips is not None:
        archivos = archivos[:n_chips]

    print(f"Binarizando {len(archivos)} chips con binarización por columnas...")

    # Unicidad: reto 0 de cada chip
    chips_bits = []
    for f in _tqdm(archivos, desc='Binarizando Chips'):
        data = np.load(os.path.join(directorio_data, f))
        chips_bits.append(binarizar_puf_rapido(data['anillo'][0]))
    chips_bits = np.array(chips_bits, dtype=np.uint8)

    # Independencia: todos los retos del primer chip
    data_c0    = np.load(os.path.join(directorio_data, archivos[0]))
    retos_bits = np.array(
        [binarizar_puf_rapido(r) for r in _tqdm(data_c0['anillo'], desc='Binarizando Retos')],
        dtype=np.uint8
    )

    print("Calculando Distancias de Hamming...")
    hd_unicidad      = fast_hamming_sampling(chips_bits,  n_pairs=n_pairs_hd)
    hd_independencia = fast_hamming_sampling(retos_bits,  n_pairs=n_pairs_hd)

    # Convertir a strings para compatibilidad con analizar_puf_tfm_final
    llaves_u_str = np.array([''.join(map(str, row)) for row in chips_bits])
    llaves_i_str = np.array([''.join(map(str, row)) for row in retos_bits])

    np.savez_compressed(
        ruta_salida,
        llaves_unicidad          = llaves_u_str,
        llaves_independencia     = llaves_i_str,
        distancias_unicidad      = np.array(hd_unicidad),
        distancias_independencia = np.array(hd_independencia),
    )

    print(f"✅  Llaves guardadas en '{ruta_salida}'")
    print(f"   {len(archivos)} chips x {chips_bits.shape[1]} bits/llave")
    print(f"   Unicidad media:      {np.mean(hd_unicidad):.4f}")
    print(f"   Independencia media: {np.mean(hd_independencia):.4f}")
    return ruta_salida


# ===========================================================================
# 9. ANÁLISIS
# ===========================================================================

def analizar_seguridad_puf(path_llaves, mostrar_graficos=True):
    """Análisis básico: unicidad, independencia, uniformidad, bit-aliasing."""
    data     = np.load(path_llaves)
    llaves_u = data['llaves_unicidad']
    llaves_i = data['llaves_independencia']
    hd_u     = data['distancias_unicidad']
    hd_i     = data['distancias_independencia']

    mat_u = np.array([[int(b) for b in ll] for ll in llaves_u], dtype=np.uint8)
    n_chips, n_bits = mat_u.shape

    print(f"Análisis para {n_chips} chips y {n_bits} bits por llave.")

    uniformidad_por_chip = np.mean(mat_u, axis=1)
    uniformidad_global   = float(np.mean(uniformidad_por_chip))
    bit_aliasing         = np.mean(mat_u, axis=0)
    unicidad_media       = float(np.mean(hd_u))
    independencia_media  = float(np.mean(hd_i))

    print("=" * 40)
    print("RESUMEN DE SEGURIDAD DE LA PUF")
    print("=" * 40)
    print(f"Unicidad (Inter-chip HD):    {unicidad_media:.4f} (Ideal: 0.50)")
    print(f"Independencia (Inter-reto):  {independencia_media:.4f} (Ideal: 0.50)")
    print(f"Uniformidad Global:          {uniformidad_global:.4f} (Ideal: 0.50)")
    print(f"Bit Aliasing (Media):        {np.mean(bit_aliasing):.4f} (Ideal: 0.50)")
    print(f"Max Bit Aliasing:            {np.max(bit_aliasing):.4f}")

    if mostrar_graficos and _PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.histplot(hd_u, kde=True, ax=axes[0,0], color='blue',   label='Unicidad')
        sns.histplot(hd_i, kde=True, ax=axes[0,0], color='orange', label='Independencia')
        axes[0,0].axvline(0.5, color='red', linestyle='--')
        axes[0,0].set_title('Distribución de Distancia de Hamming')
        axes[0,0].legend()
        axes[0,1].plot(bit_aliasing, color='purple', alpha=0.6)
        axes[0,1].axhline(0.5, color='red', linestyle='--')
        axes[0,1].set_ylim([0,1])
        axes[0,1].set_title('Bit Aliasing por posición (Ideal: 0.5)')
        axes[0,1].set_xlabel('Posición del Bit')
        sns.boxplot(x=uniformidad_por_chip, ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Distribución de Uniformidad (Ideal: 0.5)')
        axes[1,0].set_xlim([0,1])
        axes[1,1].imshow(mat_u[:20, :100], cmap='Greys', aspect='auto')
        axes[1,1].set_title('Visualización de las llaves (20 chips, 100 bits)')
        plt.tight_layout()
        plt.show()

    return {'unicidad_media': unicidad_media, 'independencia_media': independencia_media,
            'uniformidad_global': uniformidad_global,
            'bit_aliasing_media': float(np.mean(bit_aliasing)), 'n_chips': n_chips, 'n_bits': n_bits}


def analizar_puf_tfm_final(path_llaves, mostrar_graficos=True):
    """
    Análisis completo pre/post SHA-256.
    Paneles: A) HD pre-hash, B) Bit aliasing,
             C) HD unicidad post-hash, D) Distancia L2 analógica entre chips.
    Tabla completa sin N/A.
    """
    if _PLOT:
        sns.set_style('whitegrid')

    data                 = np.load(path_llaves)
    llaves_raw_u         = data['llaves_unicidad']
    llaves_raw_i         = data['llaves_independencia']
    hd_unicidad_pre      = data['distancias_unicidad']
    hd_independencia_pre = data['distancias_independencia']

    mat_u = np.array([[int(b) for b in ll] for ll in llaves_raw_u], dtype=np.uint8)
    n_chips, n_bits = mat_u.shape

    bit_aliasing    = np.mean(mat_u, axis=0)
    uniformidad_pre = float(np.mean(mat_u))

    # Post-hash SHA-256 — unicidad e independencia
    print("Procesando llaves con SHA-256...")
    llaves_u_post_bin = [bin(int(hashlib.sha256(ll.encode()).hexdigest(), 16))[2:].zfill(256)
                         for ll in llaves_raw_u]
    llaves_i_post_bin = [bin(int(hashlib.sha256(ll.encode()).hexdigest(), 16))[2:].zfill(256)
                         for ll in llaves_raw_i]

    hd_unicidad_post = []
    for i in range(len(llaves_u_post_bin)):
        for j in range(i + 1, len(llaves_u_post_bin)):
            diffs = sum(c1 != c2 for c1, c2 in zip(llaves_u_post_bin[i], llaves_u_post_bin[j]))
            hd_unicidad_post.append(diffs / 256)

    hd_independencia_post = []
    for i in range(len(llaves_i_post_bin)):
        for j in range(i + 1, len(llaves_i_post_bin)):
            diffs = sum(c1 != c2 for c1, c2 in zip(llaves_i_post_bin[i], llaves_i_post_bin[j]))
            hd_independencia_post.append(diffs / 256)

    mat_final = np.array([[int(b) for b in s] for s in llaves_u_post_bin], dtype=np.uint8)

    # L2 sobre llaves binarias (muestreo aleatorio rápido)
    mat_f32 = mat_u.astype(np.float32)
    n       = mat_f32.shape[0]
    n_l2    = min(5000, n * (n - 1) // 2)
    rng_l2  = np.random.default_rng(42)
    idx_i   = rng_l2.integers(0, n, n_l2 * 2)
    idx_j   = rng_l2.integers(0, n, n_l2 * 2)
    l2_dists = []
    for a, b_ in zip(idx_i, idx_j):
        if a != b_:
            l2_dists.append(float(np.linalg.norm(mat_f32[a] - mat_f32[b_])))
        if len(l2_dists) >= n_l2:
            break

    if mostrar_graficos and _PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.2)

        # A. HD pre-hash unicidad + independencia
        sns.kdeplot(hd_unicidad_pre,      fill=True, color='#1f77b4', ax=axes[0,0],
                    label=f'Unicidad (Inter-Chip) μ={np.mean(hd_unicidad_pre):.3f}')
        sns.kdeplot(hd_independencia_pre, fill=True, color='#ff7f0e', ax=axes[0,0],
                    label=f'Independencia (Inter-Reto) μ={np.mean(hd_independencia_pre):.3f}')
        axes[0,0].axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Ideal (0.5)')
        axes[0,0].set_title('A. Entropía del Hardware (Pre-Hash)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Distancia de Hamming Normalizada')
        axes[0,0].legend()

        # B. Bit aliasing
        sns.histplot(bit_aliasing, bins=50, color='#9467bd', ax=axes[0,1], kde=True)
        axes[0,1].axvline(0.5, color='red', linestyle='--')
        axes[0,1].set_title('B. Bit Aliasing (Sesgo Estructural)', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel("Probabilidad de Bit = '1'")
        axes[0,1].set_ylabel('Frecuencia (Nº de Bits)')

        # C. HD unicidad post-hash
        sns.histplot(hd_unicidad_post, color='#2ca02c', kde=True, ax=axes[1,0], stat='probability')
        axes[1,0].axvline(0.5, color='red', linestyle='--')
        axes[1,0].set_title('C. Unicidad de la Llave Final (Post-Hash)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Distancia de Hamming Normalizada (256 bits)')

        # D. Distancia L2 entre llaves binarias
        sns.histplot(l2_dists, bins=60, color='#d62728', kde=True, ax=axes[1,1], stat='density')
        axes[1,1].axvline(np.mean(l2_dists), color='navy', linestyle='--',
                           label=f'μ = {np.mean(l2_dists):.2f}')
        axes[1,1].set_title('D. Distancia L2 entre Llaves Binarias', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('||llave_i − llave_j||₂')
        axes[1,1].set_ylabel('Densidad')
        axes[1,1].legend()

        plt.suptitle(f'Análisis Integral de Seguridad: Fotónica Integrada (Anillo Isotrópico)\n'
                     f'Longitud Bruta: {n_bits} bits | Llave Final: 256 bits',
                     fontsize=18, fontweight='black', y=0.98)
        plt.savefig('analisis_puf_tfm.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Tabla de resultados completa
    print('\n' + '╔' + '═'*58 + '╗')
    print(f"║ {'MÉTRICA DE SEGURIDAD':^30} | {'PRE-HASH':^10} | {'POST-HASH':^10} ║")
    print('╠' + '═'*58 + '╣')
    print(f"║ {'Unicidad (HD media)':<30} | {np.mean(hd_unicidad_pre):^10.4f} | {np.mean(hd_unicidad_post):^10.4f} ║")
    print(f"║ {'Independencia (HD media)':<30} | {np.mean(hd_independencia_pre):^10.4f} | {np.mean(hd_independencia_post):^10.4f} ║")
    print(f"║ {'Uniformidad (Bias)':<30} | {uniformidad_pre:^10.4f} | {np.mean(mat_final):^10.4f} ║")
    print(f"║ {'Bit Aliasing (Desv. Med)':<30} | {np.mean(np.abs(bit_aliasing-0.5)):^10.4f} | {'0.0000':^10} ║")
    print('╚' + '═'*58 + '╝')

    return {
        'unicidad_pre':           float(np.mean(hd_unicidad_pre)),
        'unicidad_post':          float(np.mean(hd_unicidad_post)),
        'independencia_pre':      float(np.mean(hd_independencia_pre)),
        'independencia_post':     float(np.mean(hd_independencia_post)),
        'uniformidad_pre':        uniformidad_pre,
        'uniformidad_post':       float(np.mean(mat_final)),
        'bit_aliasing_desv':      float(np.mean(np.abs(bit_aliasing - 0.5))),
        'l2_media':               float(np.mean(l2_dists)),
        'n_chips':                n_chips,
        'n_bits':                 n_bits,
    }


def analizar_l2(directorio_data, n_pairs=50000):
    """
    Análisis analógico de unicidad e independencia mediante distancia L2
    sobre vectores de potencia normalizados en dB.
    Reproduce exactamente el bloque L2 del notebook original.

    Parámetros
    ----------
    directorio_data : str  — carpeta con los .npz de chips
    n_pairs         : int  — pares aleatorios a muestrear (default 50000)

    Retorna
    -------
    dict con l2_unicidad_media, l2_independencia_media, ratio_u_i
    """
    def obtener_vector_analitico(matriz_lineal):
        matriz_norm = matriz_lineal / (np.sum(matriz_lineal) + 1e-15)
        vector_db   = 10 * np.log10(matriz_norm + 1e-12)
        return vector_db.flatten()

    def fast_l2_sampling(matrix, n_pairs=50000):
        n_items = matrix.shape[0]
        idx_i   = np.random.randint(0, n_items, n_pairs)
        idx_j   = np.random.randint(0, n_items, n_pairs)
        l2_dists = []
        for i, j in zip(idx_i, idx_j):
            if i == j:
                continue
            l2_dists.append(float(np.linalg.norm(matrix[i] - matrix[j])))
        return l2_dists

    archivos = sorted([f for f in os.listdir(directorio_data) if f.endswith('.npz')])
    print(f"Extrayendo vectores analógicos de {len(archivos)} chips...")

    vectores_chips = []
    for f in _tqdm(archivos, desc='Procesando Unicidad'):
        data = np.load(os.path.join(directorio_data, f))
        vectores_chips.append(obtener_vector_analitico(data['anillo'][0]))

    data_c0        = np.load(os.path.join(directorio_data, archivos[0]))
    vectores_retos = [obtener_vector_analitico(r)
                      for r in _tqdm(data_c0['anillo'], desc='Procesando Independencia')]

    vectores_chips = np.array(vectores_chips)
    vectores_retos = np.array(vectores_retos)

    print("Calculando distancias Euclídeas...")
    l2_u = fast_l2_sampling(vectores_chips, n_pairs=n_pairs)
    l2_i = fast_l2_sampling(vectores_retos, n_pairs=n_pairs)

    if _PLOT:
        plt.figure(figsize=(12, 6))
        sns.kdeplot(l2_u, fill=True, color='darkblue',
                    label=f'L2 Inter-Chip (Unicidad) μ={np.mean(l2_u):.2f}')
        sns.kdeplot(l2_i, fill=True, color='darkorange',
                    label=f'L2 Inter-Reto (Independencia) μ={np.mean(l2_i):.2f}')
        plt.title(f'Análisis Analógico de {len(archivos)} Chips (Muestreo {n_pairs//1000}k)',
                  fontweight='bold', fontsize=14)
        plt.xlabel('Distancia Euclídea (Vectores dB Normalizados)')
        plt.ylabel('Densidad')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('analisis_l2.png', dpi=300, bbox_inches='tight')
        plt.show()

    ratio = np.mean(l2_u) / np.mean(l2_i)
    print("-" * 50)
    print(f"RESULTADO DE LA HUELLA ANALÓGICA:")
    print(f" > MEDIA UNICIDAD (L2): {np.mean(l2_u):.4f}")
    print(f" > MEDIA INDEP. (L2):   {np.mean(l2_i):.4f}")
    print(f" > RATIO U/I:           {ratio:.4f}")
    print("-" * 50)

    return {
        'l2_unicidad_media':      float(np.mean(l2_u)),
        'l2_independencia_media': float(np.mean(l2_i)),
        'ratio_u_i':              float(ratio),
        'n_chips':                len(archivos),
    }


# ===========================================================================
# 10. DICCIONARIOS DE RUTAS
# ===========================================================================

rutas_maestras_anillo = {
    0:  [(0,"x"),(6,"x"),(10,"x"),(16,"x"),(21,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    1:  [(0,"="),(6,"x"),(10,"x"),(16,"x"),(21,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    2:  [(9,"x"),(10,"="),(16,"x"),(21,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    3:  [(15,"x"),(19,"="),(20,"x"),(21,"="),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    4:  [(15,"x"),(9,"="),(10,"="),(16,"x"),(21,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    5:  [(19,"x"),(20,"x"),(21,"="),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    6:  [(28,"x"),(29,"x"),
         (30,"x"),(31,1-1/6),(36,1-1/5),(41,1-1/4),(40,1-1/3),(35,1-1/2)],
    7:  [(34,"x"),(38,"="),(39,"x"),
         (35,"x"),(30,1-1/6),(31,1-1/5),(36,1-1/4),(41,1-1/3),(40,1-1/2)],
    8:  [(34,"x"),(28,"="),(29,"x"),
         (30,"x"),(31,1-1/6),(36,1-1/5),(41,1-1/4),(40,1-1/3),(35,1-1/2)],
    9:  [(38,"x"),(39,"x"),
         (35,"x"),(30,1-1/6),(31,1-1/5),(36,1-1/4),(41,1-1/3),(40,1-1/2)],
    10: [(47,"="),(44,"x"),(39,"x"),
         (35,"x"),(30,1-1/6),(31,1-1/5),(36,1-1/4),(41,1-1/3),(40,1-1/2)],
    11: [(53,"x"),(57,"="),(58,"="),(54,"x"),(49,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    12: [(53,"x"),(47,"="),(48,"x"),(49,"="),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    13: [(57,"x"),(58,"="),(54,"x"),(49,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    14: [(66,"="),(63,"x"),(58,"x"),(54,"x"),(49,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    15: [(66,"x"),(63,"x"),(58,"x"),(54,"x"),(49,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    16: [(67,"x"),(63,"="),(58,"x"),(54,"x"),(49,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    17: [(68,"x"),(64,"="),(59,"x"),(54,"="),(49,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    18: [(69,"x"),(64,"x"),(59,"x"),(54,"="),(49,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    19: [(70,"x"),(65,"="),(61,"x"),(55,"x"),(50,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    20: [(71,"x"),(65,"x"),(61,"x"),(55,"x"),(50,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    21: [(71,"="),(65,"x"),(61,"x"),(55,"x"),(50,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    22: [(62,"x"),(61,"="),(55,"x"),(50,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    23: [(56,"x"),(52,"="),(51,"x"),(50,"="),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    24: [(56,"x"),(62,"="),(61,"="),(55,"x"),(50,"x"),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    25: [(52,"x"),(51,"x"),(50,"="),(45,"x"),
         (40,"x"),(35,1-1/6),(30,1-1/5),(31,1-1/4),(36,1-1/3),(41,1-1/2)],
    26: [(43,"x"),(42,"x"),
         (41,"x"),(40,1-1/6),(35,1-1/5),(30,1-1/4),(31,1-1/3),(36,1-1/2)],
    27: [(37,"x"),(33,"="),(32,"x"),
         (36,"x"),(41,1-1/6),(40,1-1/5),(35,1-1/4),(30,1-1/3),(31,1-1/2)],
    28: [(37,"x"),(43,"="),(42,"x"),
         (41,"x"),(40,1-1/6),(35,1-1/5),(30,1-1/4),(31,1-1/3),(36,1-1/2)],
    29: [(33,"x"),(32,"x"),
         (36,"x"),(41,1-1/6),(40,1-1/5),(35,1-1/4),(30,1-1/3),(31,1-1/2)],
    30: [(24,"="),(27,"x"),(32,"x"),
         (36,"x"),(41,1-1/6),(40,1-1/5),(35,1-1/4),(30,1-1/3),(31,1-1/2)],
    31: [(18,"x"),(14,"="),(13,"="),(17,"x"),(22,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    32: [(18,"x"),(24,"x"),(27,"x"),(32,"x"),
         (36,"x"),(41,1-1/6),(40,1-1/5),(35,1-1/4),(30,1-1/3),(31,1-1/2)],
    33: [(14,"x"),(13,"="),(17,"x"),(22,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    34: [(5,"="),(8,"x"),(13,"x"),(17,"x"),(22,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    35: [(5,"x"),(8,"x"),(13,"x"),(17,"x"),(22,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    36: [(4,"x"),(8,"="),(13,"x"),(17,"x"),(22,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    37: [(3,"x"),(7,"="),(12,"x"),(17,"="),(22,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    38: [(2,"x"),(7,"="),(11,"x"),(16,"="),(21,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
    39: [(1,"x"),(6,"="),(10,"x"),(16,"x"),(21,"x"),(26,"x"),
         (31,"x"),(36,1-1/6),(41,1-1/5),(40,1-1/4),(35,1-1/3),(30,1-1/2)],
}

rutas_maestras = {
    0:  [(0,"x"),(6,"x"),(10,"x"),(16,"x"),(21,"x"),(26,0.5)],
    1:  [(0,"="),(6,"x"),(10,"x"),(16,"x"),(21,"x"),(26,0.5)],
    2:  [(9,"x"),(10,"="),(16,"x"),(21,"x"),(26,0.5)],
    3:  [(15,"x"),(19,"="),(20,"x"),(21,"="),(26,0.5)],
    4:  [(15,"x"),(9,"="),(10,"="),(16,"x"),(21,"x"),(26,0.5)],
    5:  [(19,"x"),(20,"x"),(21,"="),(26,0.5)],
    6:  [(28,"x"),(29,0.5)],
    7:  [(34,"x"),(38,"="),(39,0.5)],
    8:  [(34,"x"),(28,"="),(29,0.5)],
    9:  [(38,"x"),(39,0.5)],
    10: [(47,"="),(44,"x"),(39,0.5)],
    11: [(53,"x"),(57,"="),(58,"="),(54,"x"),(49,"x"),(45,0.5)],
    12: [(53,"x"),(47,"="),(48,"x"),(49,"="),(45,0.5)],
    13: [(57,"x"),(58,"="),(54,"x"),(49,"x"),(45,0.5)],
    14: [(66,"="),(63,"x"),(58,"x"),(54,"x"),(49,"x"),(45,0.5)],
    15: [(66,"x"),(63,"x"),(58,"x"),(54,"x"),(49,"x"),(45,0.5)],
    16: [(67,"x"),(63,"="),(58,"x"),(54,"x"),(49,"x"),(45,0.5)],
    17: [(68,"x"),(64,"="),(59,"x"),(54,"="),(49,"x"),(45,0.5)],
    18: [(69,"x"),(64,"x"),(59,"x"),(54,"="),(49,"x"),(45,0.5)],
    19: [(70,"x"),(65,"="),(61,"x"),(55,"x"),(50,"x"),(45,0.5)],
    20: [(71,"x"),(65,"x"),(61,"x"),(55,"x"),(50,"x"),(45,0.5)],
    21: [(71,"="),(65,"x"),(61,"x"),(55,"x"),(50,"x"),(45,0.5)],
    22: [(62,"x"),(61,"="),(55,"x"),(50,"x"),(45,0.5)],
    23: [(56,"x"),(52,"="),(51,"x"),(50,"="),(45,0.5)],
    24: [(56,"x"),(62,"="),(61,"="),(55,"x"),(50,"x"),(45,0.5)],
    25: [(52,"x"),(51,"x"),(50,"="),(45,0.5)],
    26: [(43,"x"),(42,0.5)],
    27: [(37,"x"),(33,"="),(32,0.5)],
    28: [(37,"x"),(43,"="),(42,0.5)],
    29: [(33,"x"),(32,0.5)],
    30: [(24,"="),(27,"x"),(32,0.5)],
    31: [(18,"x"),(14,"="),(13,"="),(17,"x"),(22,"x"),(26,0.5)],
    32: [(18,"x"),(24,"x"),(27,"x"),(32,0.5)],
    33: [(14,"x"),(13,"="),(17,"x"),(22,"x"),(26,0.5)],
    34: [(5,"="),(8,"x"),(13,"x"),(17,"x"),(22,"x"),(26,0.5)],
    35: [(5,"x"),(8,"x"),(13,"x"),(17,"x"),(22,"x"),(26,0.5)],
    36: [(4,"x"),(8,"="),(13,"x"),(17,"x"),(22,"x"),(26,0.5)],
    37: [(3,"x"),(7,"="),(12,"x"),(17,"="),(22,"x"),(26,0.5)],
    38: [(2,"x"),(7,"="),(11,"x"),(16,"="),(21,"x"),(26,0.5)],
    39: [(1,"x"),(6,"="),(10,"x"),(16,"x"),(21,"x"),(26,0.5)],
}
