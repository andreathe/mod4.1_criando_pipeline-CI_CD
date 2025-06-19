from funcoes import matematica

def test_soma():
    assert matematica.soma(2, 3) == 5

def test_divide():
    assert matematica.divide(10, 2) == 5

def test_divide_zero():
    try:
        matematica.divide(10, 0)
        assert False
    except ValueError:
        assert True

# Crie testes para todos os casos das outras funções que você criou