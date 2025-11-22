from gtfread import read_gtf

def test_readgtf():
    df = read_gtf("tests/annotation.gtf.gz")
