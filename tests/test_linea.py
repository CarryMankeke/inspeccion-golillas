from golillas.linea import Parametros, simular


def vision_perfecta(p, rng):
    return p.mala, 30.0


def test_con_vision_perfecta_y_encoder_se_retiran_todas_las_malas():
    sim = simular(Parametros(v0=300, p_mala=0.3), "encoder+sensor", vision_perfecta, duracion=20, semilla=1)
    r = sim.resumen()
    assert r["malas"] > 10
    assert r["malas_retiradas"] == 1.0
    assert r["buenas_retiradas"] <= 0.02


def test_el_temporizador_falla_cuando_la_cinta_cambia_de_velocidad():
    par = Parametros(v0=500, p_mala=0.3, oscilacion=0.15)
    t = simular(par, "temporizador", vision_perfecta, duracion=30, semilla=2).resumen()
    e = simular(par, "encoder+sensor", vision_perfecta, duracion=30, semilla=2).resumen()
    assert e["malas_retiradas"] > t["malas_retiradas"]
