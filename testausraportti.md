# Testausraportti

Projektin lopputulosta testataan kahdella eri tavalla; yksikkötestaaminen ja neuroverkon ennustustarkkuus.

## Yksikkötestaus

Yksikkötestaamisen ajurina käytetään `pytest` moduulia. Testauskattavuus saadaan `coverage` paketilla. Alla viimeisin `coverage` raportti. `NeuralNetwork` luokka toimii fasadina ilman mitään oikeaa logiikkaa, joten sen testaaminen on jätetty pois. Myös `PersistableModel` luokka on lähinnä työkalu neuroverkon tilan tallentamiseen, eikä siten liity itse algoritmiin, joten testaaminen on jätetty pois.

```
coverage report --show-missing
Name                               Stmts   Miss  Cover   Missing
----------------------------------------------------------------
src/activation_funtions.py            33      7    79%   39, 51, 62, 73, 105, 122, 134
src/backpropagation.py                24      0   100%
src/cost_functions.py                  2      0   100%
src/sgd.py                            23      0   100%
src/test_activation_functions.py      44      0   100%
src/test_backpropagation.py           38      0   100%
src/test_sgd.py                       23      0   100%
src/test_utilities.py                 51      0   100%
src/testing_utils.py                   4      0   100%
src/utilities.py                      19      0   100%
----------------------------------------------------------------
TOTAL                                261      7    97%
```

### Testaustrategia

Testauksen kannalta mielekkäitä ohjelman osia ovat
1. vastavirta-algoritmi ([backpropagation.py](src/backpropagation.py))
1. gradienttimenetelmä ([sgd.py](src/sgd.py))
1. aktivointifunktiot ([activation_functions.py](src/activation_funtions.py))
1. apufunktiot ([utilities.py](src/utilities.py))

Varsinkin vastavirta-algoritmin ja gradienttimenetelmän osalla on käytetty funktionaalisesta ohjelmoinnista tuttua "currying"-tekniikka. Ideana on toteuttaa riippuvuuksien injektointi antamalla ne ylemmän tason funktiolle. Näin yksikkötestaamisessa voidaan käyttää yksinkertaisia, yleensä identiteetti funktiota, joiden paluuarvot ovat helposti laskettavissa ja ymmärrettävissä.

Aktivointifunktioiden ja apufunktioiden olessa luonteeltaan hyvin yksinkertaisia, menetellään niiden osalta perinteisemmin.

## Neuroverkon testaaminen

Neuroverkkoa testataan validaatio testisetillä ja testisetillä. Erona on, että validaatio testit ajetaan jokaisen opetussyklin (epoch) välissä testaamaan sen hetkistä verkon tilaa. Varsinaista testisettiä käytetään verkon opettamisen jälkeen evaluoimaan sen tarkkuutta.

Molemmat testisetit ovat kooltaan 10 000 kuvavektoria ja oikeaa vastausta.

Neuroverkoja opetetaan ja evaluoidaan [neural_network](src/neural_network.ipynb) Jupyter notebookissa. Avaamalla kyseisen tiedoston Githubissa, nähdään viimeisimman ajoni raportti.
