# Määrittelydokumentti

Tässä projektissa toteutetaan neuroverkkopohjainen numeron tunnistus algoritmi käyttäen Python ohjelmointikieltä sekä [Numpy](https://numpy.org/) kirjaston lineaarialgebra funktiota. Numpy tarjoaa Python rajapinnan tehokkaisiin C++:lla toteuttuihin apurutiinihin. Numpy on valittu juuri tehokuussyistä; vektorisoitu laskenta hyödyntämällä useampaa ydintä.

Neuroverkon osalta toteutetaan
1. Aktivointi funktio (Sigmoid / ReLU)
1. Feed forward
1. Back propagation
1. Stochastic gradient descent

En koe mielekkääksi määritellä tila ja aikavaatimuksia tälle projektille, mutta tarkoituksenani on päästä yli 98% tarkkuuteen.

Olen Senior / Lead -tason ohjelmooja yli 10 vuoden kokemuksella. Hallitsen seuraavat ohjelmointikielet: JavaScript/TypeScript, Python, C#/Java, C++, Go. Pystyn tarvittaessa katselmoimaan Rust koodia. Kuulun Tietojenkäsittelytieteiden kandidaatti ohjelmaan.

Valitsin tämän projektin, sillä haluan rakentaa neuroverkon itse niin sanotusti "pitkästä tavarasta". Jatkan opintojani Datatieteiden maisteriin, joten tämä projekti tulee antamaan hyvän pohjan tulevalle oppimiselle.

## Ohjelman toiminta

Ohjalma toteutetaan niin sanottuna CLI-ohjelmana. Ohjelmalle annetaan kuvatiedosto parametrinä ja se palauttaa numeron merkkijonona stdout-virtaan.

Mikäli aikaa riittää, tarkoituksena on antaa käyttäjän itse kirjoittamia numerosarjoja paperilla yksittäisten numeroiden sijaan.

## Dokumentaatio ja kieli

Projektin dokumentaatio mukaan lukien koodikommentit tullaan kirjoittamaan suomen kielellä. Kuitenkin koodissa olevat muuttujanimissä käytetään englantia, sillä lähtökohtaisesti kaikki ohjelmointikielet ja niiden rakenteet rakentuvan englannin päälle.

## Lähteet

Projektissa aion käyttää seuraavia lähteitä apuna:

1. Neural Networks and Deep Learning (Michael A. Nielsen, 2015)
1. Andrej Karpathy Youtube luennot.
