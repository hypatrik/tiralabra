# Helsingin yliopiston Aineopintojen harjoitustyö: Tietorakenteet ja algoritmit

![Yksikkötestit](https://github.com/hypatrik/tiralabra/workflows/Tests/badge.svg) [![codecov](https://codecov.io/gh/hypatrik/tiralabra/graph/badge.svg?token=7ZZPBK2MC3)](https://codecov.io/gh/hypatrik/tiralabra)


**Tekijä**: Patrik Keinonen

**Totetus**: 2023 loppukesä

## Dokumentit
* [Määrittelydokumentti](m%C3%A4%C3%A4rittelydokumentti.md)
* [Testausraportti](testausraportti.md)
* [Toteutusraportti](toteusraportti.md)

## Ohje

Tässä projektissa käytetään käyttöliittymänä [Jupyter notebookia](https://jupyter.org/). Suosittelen kuitenkin käyttämään VS Code pluginia, ohjeet löydät [täältä](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). Muistahan valita suoritusympäristöksi Poetryn luoman virtuaali ympäristön. Lisätietoja löydät [täältä](https://code.visualstudio.com/docs/python/environments).

Voit käyttää Jupyter notebookia [Anaconda](https://www.anaconda.com/) alustan kautta. Huomaa, että silloin joudut itse pitämään huolen, että kaikki riippuvuudet tulee asennettua.

Projektin ensijainen notebook on [neural_network.ipynb](src/neural_network.ipynb). Se pitää sisällään neuroverkon erilaisten konfiguraation kuten aktivointifunktioiden testaamisen ja raporttoinnin.

Muut `*.ipynb` ovat lisämateriaalia:
* [activation_functions.ipynb](src/activation_functions.ipynb) sisältää aktivointifunktioiden kuvaajia
* [image_transform.ipynb](src/image_transform.ipynb) on MNIST datasetin tutkimista varten

### Riipuvuudet

Riippuvuuksia halliinoidaan [Poetryllä](https://python-poetry.org/). Projektissa on käytössä [Makefile](https://opensource.com/article/18/8/what-how-makefile). Kaikki `make` komennot suoritetaan projektin juuressa.

Helpoiten asennat riippuvuudet suorittamalla:

```
> make install
```

Poetry:n virtuaalia ympäristöön (virtual env) pääset suorittamalla seuraavan komennon:

```
> make venv
```

### Yksikkötestit

Yksikkö testit käyttävät `pytest` pakettia. Helpoiten ajat yksikkötestit suorittamalla seuraavan komennon:

```
> make test
```

Komento tulostaa myös `coverage` raportin.