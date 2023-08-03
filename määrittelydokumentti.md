# Määrittelydokumentti

Tässä projektissa toteutetaan neuroverkkopohjainen numeron tunnistus algoritmi käyttäen Python ohjelmointikieltä sekä [Numpy](https://numpy.org/) kirjaston lineaarialgebra funktiota. Numpy tarjoaa Python rajapinnan tehokkaisiin C++:lla toteuttuihin apurutiinihin. Numpy on valittu juuri tehokuussyistä; vektorisoitu laskenta hyödyntämällä useampaa ydintä.

Neuroverkon osalta toteutetaan
1. Aktivointifunktiot (Sigmoid / ReLU / Step)
1. Back propagation
1. Stochastic gradient descent

Lopputulemassa tarkastellaan miten eri aktivointifunktiot vaikuttavat numerotunnistuksen tarkkutteen. Tavoitteena on myös pystyä antamaan verkon koko parametrinä.

Olen Senior-tason ohjelmooja yli 10 vuoden kokemuksella web ja pilvipohjaisesta ohjelmistokehityksestä. Hallitsen seuraavat ohjelmointikielet: JavaScript/TypeScript, Python, C#/Java, C++, Go. Pystyn tarvittaessa katselmoimaan Rust koodia. Kuulun Tietojenkäsittelytieteiden kandidaatti ohjelmaan.

Valitsin tämän projektin, sillä haluan rakentaa neuroverkon alusta alkaen itse niin sanotusti "pitkästä tavarasta". Aikaisempi kosketukseni neuroverkkoihin oli 2018 Andrew Ng Coursera koneoppimiskurssilla, missä neuroverkko toteutettiin täyttämällä osia valmiiseen pohjaan. Jatkan opintojani Datatieteiden maisteriin, joten tämä projekti tulee antamaan hyvän pohjan tulevalle oppimiselle.

## Ohjelman toiminta

Ohjelma toteutetaan Jupyter notebookiina. Kuitenkin niin, että eri osat ovat jaettu loogisiin kokonaisuuksiin omiin .py tiedostoihin yksikkötestaamisen ja ohjelman toiminnan seuraamisen helpottamiseksi. Kokonaisuuksia ovat mm. malli, aktivointifunktiot ja merkkitunnistuksen hyvyyden testaustyökalu.

Ohjelma ottaa parametriksi aktivointi funktion ja verkon koon. Jupyter notebookissa tulee olemaan omat solut eri konfiguraatioille.

MNIST datasetti sisältää 60 000 kuvaa opetukseen ja 10 000 kuvaa testaamiseen.

## Dokumentaatio ja kieli

Projektin dokumentaatio mukaan lukien koodikommentit tullaan kirjoittamaan suomen kielellä. Kuitenkin koodissa olevat muuttujanimissä käytetään englantia, sillä lähtökohtaisesti kaikki ohjelmointikielet ja niiden rakenteet rakentuvan englannin päälle.

## Lähteet

Projektissa aion käyttää seuraavia lähteitä apuna:

1. https://tim.jyu.fi/view/143092#lis%C3%A4tietoa-aktivointifunktioista
1. Neural Networks and Deep Learning (Michael A. Nielsen, 2015)
1. Andrej Karpathy Youtube luennot.
