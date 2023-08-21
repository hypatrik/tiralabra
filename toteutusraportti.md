# Toteutusraportti

Projektissa toteutetaan perintinen neuroverkko joka opetetaan tunnistmaan numeroita käyttämällä tunnettua [MNIST datasettiä](https://en.wikipedia.org/wiki/MNIST_database). Tarkoituksena on verrata neuroverkon eri aktivointifunktioiden vaikutusta tunnistamisen tarkkuuteen.

Olen kommentoinut koodia reilusti ja pyrkinyt avaamaan mahdollisimman hyvin monimutkaisimpia matemaattisia osioita sekä jättämään kommentteihin viittauksia kaavoihin ja lähteisiin. Toivottavasti lukija kokee nämä hyödyllisinä ja aihetta avaavana.

## Neuroverkko

Neuroverkon toteutus on klassinen neuroverkko ja se perustuu hyvin pitkälti [Jyväskylän yliopiston Johdatus tekoälyn matematiikkaan -kurssin monisteen kappaleeseen 2](https://tim.jyu.fi/view/143092#keinotekoiset-neuroverkot-artificial-neural-networks) sekä kirjaan Neural Networks and Deep Learning (Michael A. Nielsen, 2015).

Neuroverkkon konfiguraatiota (piilokerrokset ja aktivointifunktio) sekä hyperparametrejä pystyy säätämään. Rajapintakuvauksen löytyvät luokkien ja funktiden [pydoc](https://docs.python.org/3/library/pydoc.html) kommenteissa sekä esimerkki käytöstä [neural_network.ipnyb](src/neural_network.ipynb) Jupyter notebookissa.

### Rakenne

Neuroverkon eri osat ovat eritytetty erillisiksi kokonaisuuksiksi selkiyttämään koodia ja helpottamaan testaamista. Kokonaisuuksia voidaan testata itsenäisesti, injektoimalla riippuvuuksia testattava kokonaisuus eristetään muusta maailmasta. Tämä johtaa helposti laskettaviin lopputuloksiin, missä ei tarvitse ottaa riippuvuuksien tuomaa lisäkompleksisuutta huomioon.

Tiedostossa [model.py](src/model.py) on luokka `NeuralNetwork`. Tämä on luokka on *fasadi*  joka kokoaan yhteen edellisessä kappaleessa kuvaillut erilliset kokonaisuudet. Luokka ottaa konstruktorissa piilokerroksien konfiguraation sekä [aktivointifunktion](src/activation_functions.ipynb) parametreinä.

Vastavirta-algoritmi, joka on neuroverkon oppmisen kovaa ydintä, löytyy tiedostosta [backpropagation.py](src/backpropagation.py). Toteutus käyttää HOC (Higher order function) tyyppistä ratkaisua tehtaana. Myös Currying termillä tunnettu tekniikka mahdollistaa funktiolle elegantin riippuvuuksien injektoinnin.

Samaa tekniikkaa käyttää Stokastinen gradienttimenetelmä [sgd.py](src/sgd.py). Tiedostossa on funktiot gradienttimenetelmälle sekä sen päivitys funktiolle. Gradienttimenetelmää hyödynnetään virhefunktion [cost_functions.py](src/cost_functions.py) minimoimisessa. Gradienttimenetelmä tarvitsee virhefunktion osittaisderivaatat, jotka lasketaan vastavirta-algoritmillä. Lisää aiheesta [täällä](https://tim.jyu.fi/view/143092#neuroverkon-opettaminen).

### Opetus

Opetus tapahtuu `NeuralNetwork` luokan `fit` instanssimetodilla. Jokainen opetuskierros (epoch) palauttaa validaatiosettiä vasten ajetun testitarkkuuden. Metodi palauttaa listan tupleja (epoch, tarkkuus) opetuksen edistymisen visualisointia varten.

## Aktivointifunktioiden vertailu

Aktivointifunktioden vertailu on tehty [neural_network.ipynb](src/neural_network.ipynb) Jupyter notebookissa. Notebook aukeaa renderöitynä tulosten kanssa Githubissa. Notebook pitää sisällään eri aktivointifunktioden suorituskykyvertailut.

## Työn mahdolliset puutteet ja parannusehdotukset

Mielestäni työni laajuus on kurssille sopiva. Paras tulokseni on 0.9646. Mikäli tätä halutaan parantaa ja kurssi olisi esim 10op laajuinen, olisi jatkokehitettävää hyvinkin paljon. Esim:
1. opettamisen lopettaminen riittävän aikaisin (early stopping)
1. neuroneiden osittainen poistaminen verkosta (dropout layer)
1. painojen pienentämien L1- ja L2- säännöstelyllä (regularization)
1. opetusdatan keinotekoinen laajentaminen
1. convolutional neural network -arkkitehtuuri

Edellämainuttuja tekniikoita päästäisiin jo huomattavasti korkeampaan tarkkuutteen, jopa 99%.