"""
The tests in this script were generated in conjunction with Claude Sonnet 5 but were modified and verified by the author (Nataly Panczyk).
"""

import pytest
from pke import HolosPK


@pytest.fixture
def reactor():
    return HolosPK()


def test_calc_reactivity_at_ss(reactor):
    state = reactor.get_initial_conditions()
    ss_drum_angles = [reactor.u0] * 8
    rho = reactor.calc_reactivity(state, ss_drum_angles)
    assert rho == pytest.approx(0, abs=1e-16)


def test_calc_reactivity_Tf_inc(reactor):
    state = reactor.get_initial_conditions()
    state[7] += 10
    ss_drum_angles = [reactor.u0] * 8
    rho = reactor.calc_reactivity(state, ss_drum_angles)
    assert rho < 0


def test_calc_reactivity_Tf_dec(reactor):
    state = reactor.get_initial_conditions()
    state[7] += -10
    ss_drum_angles = [reactor.u0] * 8
    rho = reactor.calc_reactivity(state, ss_drum_angles)
    assert rho > 0


def test_drum_forcing(reactor):
    # start drums at 0 degrees
    drum_angles = [0] * 8
    # rotate 90 degrees
    drum_action = [90] * 8
    # interpolate this change for a 5 second interval
    forcer_funcs = reactor.drum_forcing(drum_angles, drum_action, time=5)
    # a 1d interpolation should yield: theta = 18*t
    # check that at 2 seconds, drums have rotated 36 degrees
    assert forcer_funcs[0](2) == 36
