"""
smartlight_puf_ipronics.py
==========================
Adaptador del pipeline PUF para el simulador oficial de iPronics.

Sustituye el Gemelo Digital (smartlight_puf_v2.py) por el simulador
oficial SmartLight de iPronics, manteniendo exactamente el mismo
formato de salida (.npz) y el mismo pipeline de binarización y análisis.

Requisitos
----------
- smartlight instalado (pip install smartlight o desde iPronics)
- wupiengine con licencia activa
- Fichero de configuración: hw_config_virtual_amf_s.smartlight

Uso básico
----------
    from smartlight_puf_ipronics import ejecutar_simulacion_ipronics
    from smartlight_puf_v2 import (
        generar_llaves_npz,
        analizar_puf_tfm_final,
        rutas_maestras_anillo,
    )

    ejecutar_simulacion_ipronics(
        config_path='hw_config_virtual_amf_s.smartlight',
        n_chips=10,
        n_retos=10,
        directorio='db_ipronics'
    )
    generar_llaves_npz('db_ipronics')
    analizar_puf_tfm_final('llaves_binarizadas_lehmer_columnas.npz')
"""

import os
import numpy as np

# Puertos activos del chip SmartLight AMF-S (28 de 40)
PUERTOS_ACTIVOS_IPRONICS = [i for i in range(40) if i < 22 or i > 33]


# ===========================================================================
# Funciones de configuración de fases pasivas (ADN del chip)
# ===========================================================================

def _aplicar_fases_pasivas_aleatorias(sl, seed: int) -> None:
    """
    Simula el ADN del chip aplicando fases pasivas de fabricación aleatorias.
    Usa set_passive_phase_design directamente en cada PUC del WMA,
    que es el método correcto para modificar el passive_phase sin tocar
    las driven_phases.

    Parámetros
    ----------
    sl   : instancia de Smartlight (simulation=True)
    seed : semilla para reproducibilidad
    """
    rng = np.random.default_rng(seed)
    wma = sl.get_wma()
    delta_phi = 2 * np.pi / 256

    for puc in wma.pucs:
        fase = rng.uniform(0, 2 * np.pi)
        fase_q = round(fase / delta_phi) * delta_phi
        puc.set_passive_phase_design(fase_q)


def _estado_a_fases(estado) -> tuple:
    """
    Convierte un estado de la ruta ('x', '=', float) a fases driven (phi1, phi2).
    Equivalente a la lógica de programar_malla en smartlight_puf_v2.

    'x'   = CROSS  -> phi1=0,   phi2=0
    '='   = BAR    -> phi1=pi,  phi2=0
    float  = K      -> phi1 = 2*arcsin(sqrt(K)), phi2=0
    """
    if estado == "x":
        return (0.0, 0.0)
    elif estado == "=":
        return (np.pi, 0.0)
    elif isinstance(estado, (int, float)):
        return (2 * np.arcsin(np.sqrt(float(estado))), 0.0)
    else:
        return (float(estado[0]), float(estado[1]))


def _generar_fases_reto(seed: int, n_pucs: int, tbus_fijas: set) -> dict:
    """
    Genera fases driven aleatorias para las TBUs dinámicas (no fijas).
    Estas fases se SUMAN a la fase pasiva del PUC en _aplicar_reto_con_ruta.
    Devuelve {puc_id: (phi1, phi2)} cuantizadas a 8 bits.
    """
    rng = np.random.default_rng(seed)
    delta_phi = 2 * np.pi / 256
    fases = {}
    for puc_id in range(n_pucs):
        if puc_id not in tbus_fijas:
            f1 = round(rng.uniform(0, 2*np.pi) / delta_phi) * delta_phi
            f2 = round(rng.uniform(0, 2*np.pi) / delta_phi) * delta_phi
            fases[puc_id] = (f1, f2)
    return fases


def _aplicar_reto_con_ruta(sl, fases_reto: dict, ruta_activa: list) -> None:
    """
    Programa las driven_phases directamente en los PUCs del WMA.

    Lógica equivalente a programar_malla de smartlight_puf_v2:
    - TBUs dinámicas: driven = phi_reto (fase total = passive + phi_reto)
    - TBUs de la ruta (fijas): driven = phi_ruta - passive
      (fase total = passive + driven = phi_ruta exacto, sin ruido de reto)

    No usa sl.set_driven_phases() ni sl.interconnect() para evitar
    que el SDK sobreescriba passive_phase con su lógica de calibración.

    Parámetros
    ----------
    sl          : instancia de Smartlight (simulation=True)
    fases_reto  : dict {puc_id: (phi1, phi2)} para TBUs dinámicas
    ruta_activa : list de (puc_id, estado) de la ruta del puerto actual
    """
    wma = sl.get_wma()

    # 1. TBUs dinámicas: driven = phi_reto
    #    fase total en _extraer_params_sdk = driven + passive = phi_reto + passive
    for puc_id, (phi1, phi2) in fases_reto.items():
        wma.pucs[puc_id].driven_phases = np.array([phi1, phi2])

    # 2. TBUs de la ruta: driven = phi_ruta - passive
    #    fase total en _extraer_params_sdk = driven + passive = phi_ruta
    for puc_id, estado in ruta_activa:
        phi_ruta, phi2_ruta = _estado_a_fases(estado)
        passive = wma.pucs[puc_id].passive_phase
        wma.pucs[puc_id].driven_phases = np.array([
            phi_ruta - passive,
            phi2_ruta
        ])


# ===========================================================================
# Motor rápido: extrae parámetros del SDK y usa el solver SMM de smartlight_puf
# ===========================================================================

def _extraer_params_sdk(sl) -> tuple:
    """
    Extrae los parámetros físicos de todos los PUCs del chip virtual.
    Devuelve (params_tbus, fases_totales, mapa_tbus) listos para usar
    con el solver SMM de smartlight_puf_v2/v3.

    params_tbus  : dict {(col, fila): {K_a, K_b, gamma_a, gamma_b, gamma_c, gamma_d}}
    fases_totales: dict {(col, fila): (phi1_total, phi2_total)}
    mapa_tbus    : dict {idx_lineal: (col, fila)}
    """
    wma = sl.get_wma()
    wl = np.array([1550e-9])
    tbus_por_columna = wma.cells_per_column
    Filas = max(tbus_por_columna)

    params_tbus   = {}
    fases_totales = {}
    mapa_tbus     = {}

    idx_lineal = 0
    for c, n_filas in enumerate(tbus_por_columna):
        for f in range(n_filas):
            puc = wma.pucs[idx_lineal]
            coords = (c, f)

            Ka = float(puc.couplers[0].get_coupling_factor(wl)[0])
            Kb = float(puc.couplers[1].get_coupling_factor(wl)[0])
            # get_insertion_loss() devuelve dB — convertir a fracción de potencia
            ga = 1 - 10**(-float(puc.couplers[0].get_insertion_loss()) / 10)
            gb = 1 - 10**(-float(puc.couplers[1].get_insertion_loss()) / 10)

            params_tbus[coords] = {
                'K_a': Ka, 'K_b': Kb,
                'gamma_a': ga, 'gamma_b': gb,
                'gamma_c': 0.0, 'gamma_d': 0.0,
            }
            # phi total = driven + passive (ya suma la fase pasiva de fabricación)
            fases_totales[coords] = (
                float(puc.driven_phases[0] + puc.passive_phase),
                float(puc.driven_phases[1]),
            )
            mapa_tbus[idx_lineal] = coords
            idx_lineal += 1

    return params_tbus, fases_totales, mapa_tbus


def _calcular_H_rapido(sl, ids_activos: list) -> np.ndarray:
    """
    Calcula la matriz de scattering usando el solver SMM de smartlight_puf.
    Es ~100x más rápido que compute_scattering_PUCs del SDK.

    Lee los parámetros físicos actuales de los PUCs (incluyendo fases driven
    y pasivas ya aplicadas) y resuelve el sistema con factorización LU dispersa.

    Parámetros
    ----------
    sl          : instancia de Smartlight (simulation=True)
    ids_activos : lista de índices de puertos perimetrales activos (0-39)

    Retorna
    -------
    np.ndarray de shape (N, N) con |S|^2 (potencias lineales)
    donde N = len(ids_activos)
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import factorized
    from smartlight_puf_v3 import TBU_SNM, malla_hexagonal

    wma = sl.get_wma()
    tbus_por_columna = wma.cells_per_column
    Filas = max(tbus_por_columna)

    # Extraer parámetros actuales del SDK
    params_tbus, fases_totales, _ = _extraer_params_sdk(sl)

    # Construir la malla base (solo geometría/cables)
    malla_base = malla_hexagonal(tbus_por_columna)
    num_puertos = malla_base.num_puertos
    num_nodos   = num_puertos * 2

    def get_nodos(p_id):
        return p_id * 2, p_id * 2 + 1

    def get_ids(c, f):
        base = (c * Filas + f) * 4
        return base, base+1, base+2, base+3

    # Construir grafo G
    G = sp.lil_matrix((num_nodos, num_nodos), dtype=complex)

    # 1. Conexiones internas TBU
    for (c, f), p in params_tbus.items():
        phi1, phi2 = fases_totales[(c, f)]
        S_tbu = TBU_SNM(p, phi1, phi2)
        ids_p = get_ids(c, f)
        for i_local in range(4):
            for j_local in range(4):
                v = S_tbu[j_local, i_local]
                if v != 0:
                    n_in  = get_nodos(ids_p[i_local])[0]
                    n_out = get_nodos(ids_p[j_local])[1]
                    G[n_out, n_in] = v

    # 2. Conexiones de cables
    malla_base._conectar_cables()
    conn = malla_base.S.tocoo()
    for p1, p2 in zip(conn.row, conn.col):
        if p1 < p2:
            n1_in, n1_out = get_nodos(p1)
            n2_in, n2_out = get_nodos(p2)
            G[n2_in, n1_out] = 1.0
            G[n1_in, n2_out] = 1.0

    # Construir mapping: puerto malla (0-39) -> nodo interno del grafo
    # ids_40 da los nodos internos en el orden de los puertos perimetrales
    from smartlight_puf_v3 import malla_SNM_completa
    malla_ref = malla_SNM_completa(tbus_por_columna, seed=None)
    ids_40 = malla_ref.obtener_ids_perimetrales_ordenados()
    # ids_40[i] = nodo interno del puerto perimetral i (en orden 0-39)
    # Necesitamos el mapping inverso: puerto_malla -> nodo_interno
    # El puerto malla i corresponde a ids_40[i]
    puerto_a_nodo = {i: ids_40[i] for i in range(len(ids_40))}

    # 3. Resolver con factorización LU
    I     = sp.eye(num_nodos, format='csr')
    A     = (I - G.tocsr()).tocsc()
    solve = factorized(A)

    N = len(ids_activos)
    S_total = np.zeros((N, N), dtype=np.float32)
    for j, p_ext in enumerate(ids_activos):
        b = np.zeros(num_nodos, dtype=complex)
        nodo_in = get_nodos(puerto_a_nodo[p_ext])[0]
        b[nodo_in] = 1.0
        x = solve(b)
        for i, p_out in enumerate(ids_activos):
            nodo_out = get_nodos(puerto_a_nodo[p_out])[1]
            S_total[i, j] = float(abs(x[nodo_out])**2)

    return S_total


# 2 wavelengths requeridas — bug del SDK: renumber_ports accede a HL[1]
_WL = np.array([1550e-9, 1550.001e-9])


def _calcular_scattering_matrix(sl) -> np.ndarray:
    """
    Calcula la matriz de scattering completa de la malla WMA.
    Devuelve H de shape (40, 40) en amplitud lineal |S|.

    Requiere 2 wavelengths para evitar el bug de renumber_ports del SDK
    (accede a HL[1] asumiendo n_wl >= 2). Toma solo la primera wavelength.

    El mapping dict_mapping_pucs_method traduce índices internos del método
    inductivo a puertos perimetrales (0-39). Se calcula solo la primera vez
    y se reutiliza en llamadas posteriores.
    """
    wma = sl.get_wma()
    wma.set_wavelengths(_WL)
    H_raw, dict_map_ports, dict_map_4ports = wma.compute_scattering_PUCs(
        wavelengths=_WL
    )
    # H_raw shape: (2, N_internal, N_internal)

    # Construir mapping puerto_malla -> índice_interno si no existe aún.
    # Se guarda en el PP (que sí tiene el atributo) para reutilizarlo.
    pp = sl.get_photonic_processor()
    if pp.dict_mapping_pucs_method is None:
        from iPronics.utilities.Scalable_Inductive_Method_PUCs.create_automatic_mesh_mapping import (
            automatic_mesh_mapping,
        )
        from iPronics.utilities.Scalable_Inductive_Method_PUCs.geometric_mapping import (
            geometric_mapping_of_the_mesh,
        )
        puc_fp, puc_pk = wma._get_info_ports_from_inductive_port_naming(
            dict_map_ports
        )
        key_and_coord = geometric_mapping_of_the_mesh(
            wma, dict_map_ports, dict_map_4ports, puc_fp, puc_pk
        )
        _, dict_final_mapping = automatic_mesh_mapping(
            wma, dict_map_ports, key_and_coord
        )
        pp.dict_mapping_pucs_method = dict_final_mapping

    # mapping: {puerto_malla(0-39): índice_interno}
    mapping = pp.dict_mapping_pucs_method
    N = wma.Nports  # 40
    H_out = np.zeros((N, N), dtype=np.float32)
    for p_out in range(N):
        for p_in in range(N):
            i_out = mapping.get(p_out)
            i_in  = mapping.get(p_in)
            if i_out is not None and i_in is not None:
                H_out[p_out, p_in] = np.abs(H_raw[0, i_out, i_in])
    return H_out  # shape (40, 40), amplitud lineal


def _medir_potencias_puerto(sl, puerto_in: int,
                             puertos_out: list,
                             H: np.ndarray = None) -> np.ndarray:
    """
    Extrae potencias de la matriz de scattering pre-calculada.

    Parámetros
    ----------
    sl          : instancia de Smartlight (simulation=True)
    puerto_in   : puerto de inyección (numeración de la malla, 0-39)
    puertos_out : lista de puertos de salida (numeración de la malla, 0-39)
    H           : matriz (40x40, amplitud lineal). Si None se calcula.

    Retorna
    -------
    np.ndarray de shape (len(puertos_out),) con potencias lineales |S|^2
    """
    if H is None:
        H = _calcular_scattering_matrix(sl)

    vector = np.array([
        H[p, puerto_in] ** 2
        for p in puertos_out
    ], dtype=np.float32)
    return vector


# ===========================================================================
# Función principal: simular un chip completo
# ===========================================================================

def _simular_chip_con_sl(sl, chip_idx: int,
                          n_retos: int,
                          rutas_anillo: dict,
                          directorio: str) -> None:
    """
    Simula un chip completo reutilizando una instancia ya creada de Smartlight.
    Cambia las fases pasivas para simular un chip distinto en cada llamada.

    Parámetros
    ----------
    sl          : instancia activa de Smartlight (simulation=True)
    chip_idx    : índice del chip (determina la semilla)
    n_retos     : número de retos por chip
    rutas_anillo: dict con las rutas del Anillo Isotrópico
    directorio  : carpeta donde guardar los .npz
    """
    seed_hw = 5000 + chip_idx
    os.makedirs(directorio, exist_ok=True)
    filename = os.path.join(directorio, f'chip_{seed_hw}.npz')
    if os.path.exists(filename):
        return

    puertos_activos = PUERTOS_ACTIVOS_IPRONICS  # 28 puertos
    N = len(puertos_activos)

    # --- ADN del chip: fases pasivas aleatorias únicas por chip ---
    _aplicar_fases_pasivas_aleatorias(sl, seed=seed_hw)

    def _medir_todos_puertos(seed_reto: int, seed_offset: int) -> np.ndarray:
        """
        Para un reto dado, calcula la matriz N×N de potencias completa.

        Optimización: para cada puerto de entrada se construye G una sola vez
        y se resuelven todos los puertos de salida con una factorización LU.
        Esto evita reconstruir G 28 veces por reto — solo se construye 1 vez
        por puerto de entrada (28 veces por reto, no 28x28).

        El fondo aleatorio es el mismo para todos los puertos de un reto
        (misma seed), y solo cambia la ruta del Anillo por puerto.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import factorized
        from smartlight_puf_v3 import TBU_SNM, malla_hexagonal

        wma = sl.get_wma()
        n_pucs = len(wma.pucs)
        tbus_por_columna = wma.cells_per_column
        Filas = max(tbus_por_columna)

        # Construir cables una sola vez (no cambian entre puertos)
        malla_base = malla_hexagonal(tbus_por_columna)
        malla_base._conectar_cables()
        conn = malla_base.S.tocoo()
        num_puertos = malla_base.num_puertos
        num_nodos   = num_puertos * 2

        def get_nodos(p_id): return p_id * 2, p_id * 2 + 1
        def get_ids(c, f):
            base = (c * Filas + f) * 4
            return base, base+1, base+2, base+3

        # Generar fondo aleatorio base (misma seed para todos los puertos)
        rng_base = np.random.default_rng(seed_reto + seed_offset)
        delta_phi = 2 * np.pi / 256
        fases_fondo = {}
        for puc_id in range(n_pucs):
            f1 = round(rng_base.uniform(0, 2*np.pi) / delta_phi) * delta_phi
            f2 = round(rng_base.uniform(0, 2*np.pi) / delta_phi) * delta_phi
            fases_fondo[puc_id] = (f1, f2)

        # Extraer params físicos del chip (K, gamma) — no cambian en el reto
        wl = np.array([1550e-9])
        params_chip = {}
        idx_lineal = 0
        for c, n_filas in enumerate(tbus_por_columna):
            for f in range(n_filas):
                puc = wma.pucs[idx_lineal]
                params_chip[(c, f)] = {
                    'K_a':     float(puc.couplers[0].get_coupling_factor(wl)[0]),
                    'K_b':     float(puc.couplers[1].get_coupling_factor(wl)[0]),
                    # get_insertion_loss() devuelve dB — convertir a fracción de potencia
                    'gamma_a': 1 - 10**(-float(puc.couplers[0].get_insertion_loss()) / 10),
                    'gamma_b': 1 - 10**(-float(puc.couplers[1].get_insertion_loss()) / 10),
                    'gamma_c': 0.0, 'gamma_d': 0.0,
                    'passive': float(puc.passive_phase),
                    'idx':     idx_lineal,
                }
                idx_lineal += 1

        # Mapa índice lineal -> (col, fila)
        mapa_lineal_a_coords = {}
        idx_tmp = 0
        for c, n_filas in enumerate(tbus_por_columna):
            for f in range(n_filas):
                mapa_lineal_a_coords[idx_tmp] = (c, f)
                idx_tmp += 1

        # Pre-calcular S_tbu para todas las TBUs con fondo aleatorio
        # Esto se reutiliza para todos los puertos del reto
        S_tbus_fondo = {}
        idx_lineal = 0
        for c, n_filas in enumerate(tbus_por_columna):
            for f in range(n_filas):
                p = params_chip[(c, f)]
                passive = p['passive']
                phi1 = passive + fases_fondo[idx_lineal][0]
                phi2 = fases_fondo[idx_lineal][1]
                S_tbus_fondo[idx_lineal] = (TBU_SNM(p, phi1, phi2), get_ids(c, f), p)
                idx_lineal += 1

        # Construir G base con fondo aleatorio (sin ninguna ruta)
        def _construir_G_base():
            G = sp.lil_matrix((num_nodos, num_nodos), dtype=complex)
            for idx, (S_tbu, ids_p, _) in S_tbus_fondo.items():
                for i_local in range(4):
                    for j_local in range(4):
                        v = S_tbu[j_local, i_local]
                        if v != 0:
                            G[get_nodos(ids_p[i_local])[0],
                              get_nodos(ids_p[j_local])[1]] = 0  # placeholder
                            n_in  = get_nodos(ids_p[i_local])[0]
                            n_out = get_nodos(ids_p[j_local])[1]
                            G[n_out, n_in] = v
            for p1, p2 in zip(conn.row, conn.col):
                if p1 < p2:
                    G[get_nodos(p2)[0], get_nodos(p1)[1]] = 1.0
                    G[get_nodos(p1)[0], get_nodos(p2)[1]] = 1.0
            return G

        G_base = _construir_G_base()
        matriz = np.zeros((N, N), dtype=np.float32)

        for idx_col, p_in in enumerate(puertos_activos):
            ruta_puerto = rutas_anillo.get(p_in, [])
            ruta_dict = {puc_id: estado for puc_id, estado in ruta_puerto}

            # Copiar G_base y sobreescribir solo las TBUs de la ruta
            G = G_base.copy()
            for puc_id, estado in ruta_puerto:
                # puc_id es índice lineal -> buscar coords (c,f)
                coords = mapa_lineal_a_coords[puc_id]
                p = params_chip[coords]
                ids_p = S_tbus_fondo[puc_id][1]
                phi1, phi2 = _estado_a_fases(estado)
                S_tbu = TBU_SNM(p, phi1, phi2)
                for i_local in range(4):
                    for j_local in range(4):
                        n_in  = get_nodos(ids_p[i_local])[0]
                        n_out = get_nodos(ids_p[j_local])[1]
                        G[n_out, n_in] = S_tbu[j_local, i_local]

            # Factorizar LU y resolver
            I     = sp.eye(num_nodos, format='csr')
            A     = (I - G.tocsr()).tocsc()
            solve = factorized(A)

            b = np.zeros(num_nodos, dtype=complex)
            b[get_nodos(p_in)[0]] = 1.0
            x = solve(b)

            for idx_fila, p_out in enumerate(puertos_activos):
                if p_out != p_in:
                    matriz[idx_fila, idx_col] = float(
                        abs(x[get_nodos(p_out)[1]])**2
                    )

        return matriz

    # --- Medida base ---
    res_base_anillo = _medir_todos_puertos(seed_reto=seed_hw + 777, seed_offset=0)

    # --- Retos aleatorios ---
    retos_anillo = []
    for r in range(n_retos):
        m_reto = _medir_todos_puertos(seed_reto=seed_hw + r + 1, seed_offset=0)
        retos_anillo.append(m_reto)

    # Guardar en formato idéntico a smartlight_puf_v2
    np.savez_compressed(
        filename,
        base_anillo=res_base_anillo,
        anillo=np.array(retos_anillo),
        seed_hw=np.array(seed_hw)
    )


# ===========================================================================
# Función de entrada principal
# ===========================================================================

def ejecutar_simulacion_ipronics(config_path: str,
                                  n_chips: int = 10,
                                  n_retos: int = 10,
                                  rutas_anillo: dict = None,
                                  directorio: str = 'db_ipronics') -> str:
    """
    Ejecuta la simulación PUF completa usando el simulador oficial iPronics.

    El formato de salida es idéntico al de ejecutar_simulacion_masiva en
    smartlight_puf_v2, por lo que el resto del pipeline (binarización,
    análisis) funciona sin cambios.

    Parámetros
    ----------
    config_path  : str  — ruta al fichero .smartlight de configuración virtual
                          (e.g. 'hw_config_virtual_amf_s.smartlight')
    n_chips      : int  — número de chips a simular
    n_retos      : int  — número de retos por chip
    rutas_anillo : dict — rutas_maestras_anillo de smartlight_puf_v2
                          (si None se importa automáticamente)
    directorio   : str  — carpeta de salida para los .npz

    Retorna
    -------
    str — ruta al directorio de salida

    Ejemplo
    -------
    from smartlight_puf_ipronics import ejecutar_simulacion_ipronics
    from smartlight_puf_v2 import (
        generar_llaves_npz,
        analizar_puf_tfm_final,
        rutas_maestras_anillo,
    )

    ejecutar_simulacion_ipronics(
        config_path='hw_config_virtual_amf_s.smartlight',
        n_chips=5,
        n_retos=5,
        rutas_anillo=rutas_maestras_anillo,
        directorio='db_ipronics_test'
    )
    generar_llaves_npz('db_ipronics_test')
    analizar_puf_tfm_final('llaves_binarizadas_lehmer_columnas.npz')
    """
    from smartlight.smartlight import Smartlight

    if rutas_anillo is None:
        from smartlight_puf_v2 import rutas_maestras_anillo
        rutas_anillo = rutas_maestras_anillo

    print(f"Simulación iPronics: {n_chips} chips x {n_retos} retos")
    print(f"Config             : {config_path}")
    print(f"Puertos activos    : {len(PUERTOS_ACTIVOS_IPRONICS)} "
          f"({PUERTOS_ACTIVOS_IPRONICS[:4]}...{PUERTOS_ACTIVOS_IPRONICS[-4:]})")
    print(f"Directorio salida  : {directorio}/")
    print()

    # Crear el chip UNA SOLA VEZ y reutilizarlo cambiando las fases pasivas.
    # Esto evita el error de singleton de Smartlight.
    print("Inicializando chip virtual...", end=" ", flush=True)
    sl = Smartlight(config_path, simulation=True)
    print("OK\n")

    try:
        for chip_idx in range(n_chips):
            seed_hw = 5000 + chip_idx
            filename = os.path.join(directorio, f"chip_{seed_hw}.npz")
            if os.path.exists(filename):
                print(f"  Chip {chip_idx+1:>4}/{n_chips} — ya existe, saltando")
                continue

            print(f"  Chip {chip_idx+1:>4}/{n_chips} (seed={seed_hw})...",
                  end=" ", flush=True)
            try:
                _simular_chip_con_sl(
                    sl=sl,
                    chip_idx=chip_idx,
                    n_retos=n_retos,
                    rutas_anillo=rutas_anillo,
                    directorio=directorio,
                )
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
    finally:
        sl.disconnect()

    print(f"\n✅  Simulación iPronics completada en '{directorio}/'")
    return directorio


# ===========================================================================
# Test rápido de verificación
# ===========================================================================

def test_conexion(config_path: str) -> None:
    """
    Verifica que el chip virtual se carga correctamente y hace una medida
    de prueba. Útil para comprobar que todo funciona antes de lanzar la
    simulación completa.

    Parámetros
    ----------
    config_path : ruta al fichero .smartlight
    """
    from smartlight.smartlight import Smartlight

    print("=" * 55)
    print("TEST DE CONEXIÓN — SmartLight iPronics Virtual")
    print("=" * 55)

    sl = Smartlight(config_path, simulation=True)
    try:
        wma = sl.get_wma()
        puertos = sl.get_wma_access_ports()

        print(f"✅  Chip cargado correctamente")
        print(f"    PUCs            : {len(wma.pucs)}")
        print(f"    Puertos totales : {wma.Nports}")
        print(f"    Puertos activos : {len(puertos)}")
        print(f"    cells_per_column: {wma.cells_per_column}")
        print()

        # Medida de prueba: inyectar por puerto 0 y leer todos
        print("Medida de prueba (puerto 0 → todos)...")
        from smartlight_puf_v2 import rutas_maestras_anillo
        sl.interconnect(rutas_maestras_anillo[0])
        resultado = sl.get_output_power(inport=0, outport=puertos)
        potencias_dbm = [resultado.get(p, None) for p in puertos[:5]]
        print(f"    Primeras 5 potencias (dBm): {[f'{p:.2f}' if p else 'None' for p in potencias_dbm]}")
        print()
        print("✅  Test completado — el simulador funciona correctamente")

    finally:
        sl.disconnect()
    print("=" * 55)
