"""RAWS rows with 7-letter SHEF codes (Colorado DCP feeds) read like the 6-letter ones."""
from firesim.weather import read_station


def test_seven_letter_codes(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("station,utc_valid,TAIRGZZ,UDIRGZZ,USIRGZZ,XRIRGZZ\nX,2021-12-30 18:25:00,40.0,270.0,20.0,15.0\n")
    (t, sp, d, temp, rh, stick), = read_station(str(p), "raws")
    assert abs(sp - 20 * 0.44704) < 1e-6 and d == 270.0 and temp == 40.0 and rh == 15.0 and stick != stick
