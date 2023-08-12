# Viikkoraportti 4

Tämä viikko oli intensiivinen. Toteutin liudan aktivointifunktioita kokeiltavaksi neuroverkon kanssa; ReLU, Leaky ReLU, Tanh. Tanh-funktiolle annoin skaalaus parametrin, koska kiinnosti vertailla sitä suoraan sigmoidiin.

Ohjelmakoodia on refaktoroitu testaamisen ja käytettävyyden helpottamiseksi. Aktivointifunktioista tein luokkia, jotta alpha-parametrit ovat helpompi antaa. Nyt neuroverkolle aktivointifunktion nimen sijaan annetaan luokan instanssi. Näin jälkikäteen ajateltuna se tuntuu muutenkin tyylikkäämältä.

Edellisen viikon kysymykseeni viitaten, tein nyt yksikkötestit niille osa-alueille, joita on mielekästä testata; vastavirta-algoritmi, gradienttimenetelmä, apufunktiot ja aktivointifunktiot. Esimerkiksi `NeuralNetwork` luokan olessa fasadi edellä mainituille loogisille kokonaisuuksille, en koe mielekkääksi sen testaamista. Lisäarvo voisi olla, että parametrit mapataan oikein; toisaalta mahdolliset virheet käyvät ilmi ohjelmaa ajaettaessa.

Viikko oli hyvin opettavainen aktivointifunktioista ja hyperparametrien säädöstä. ReLU:n kanssa päädyin jatkuvasti tilanteeseen, jossa ensimmäisen opetussyklin jälkeen päädyin tilanteeseen, jossa validaatiotesti antaa tuloksesi 991/10000. Sama luku toistui jokaisen opetussyklin jälkeen. Tämä pakotti minut syventymään ReLU:n toimintaan ja opin, että neuronit kuolivat verkossani. Säätämällä `learnin_rate` parametriä huomattavasti pienemmäksi päädyin sain parempia tuloksia. Leaky ReLU aktivoinnilla sain parhaat tulokset kaikista. 

Jyväskylän yliopiston koneoppimismateriaali on ollut oikea kultakaivos vastavirta-algorimin, ReLU (ja Leaky ReLU) sekä ristientropia virhefunktion osalta. Samoin Michel Nielsenin kirja on ollut erittäin opettava. Ensi viikolla aion vielä kokeilla regulaatiotekniikoita, jos saisin 96.5% tarkkuutta vielä ylös!

Kysymys: Olen käyttänyt hyvin paljon tunteja ja saanut alkuperäisen määrittelydokumentaation mukaisen toteutuksen valmiiksi. Myös suorituskykytestausta on tehty [neural_network.ipynb](src/neural_network.ipynb) notebookissa. Voiko nyt ottaa rauhallisemmin ja kirjoittaa vain viikkoraportit + vertaisarvointi kommentit ja korjaukset? Luonnollisesti vielä dokumentaatiot pitää kirjoittaa loppuun.
