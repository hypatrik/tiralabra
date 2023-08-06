# Viikkoraportti 3

Tällä viikolla saatiin repositorioon asennettua lintteri (flake8), formatteri (black) ja coverage pytestiin. Pakentinhallintaan käytetään poetryä. Myös tehty Makefile helpottamaan komentojen käyttöä.

Neuroverkon osalta vastavirta-algoritmi eli backpropagation saatiin toimimaan kolmen yrittämän jälkeen. Materiaalina käytettiin Jyväskylän Yliopiston Koneoppmisen matematiikan perusteet kurssin materiaalia neuroverkoista. Viimeisessä vaiheessa, eli taaksepäin kulkemisessa oli ongelmia, sillä matriiseihin ilmestyi muutaman iteraation jälkeen ylimääräisiä dimensioita. Lainasin Michael A. Nielsen kirjasta ideaa käyttää Pythonin kielen ominaisuutta käyttää negatiivisia indeksejä taulukossa ja ongelma häivisi. Paljon aikaa meni tutkia matriisien kokoja, että saatiin pistetulot yms toimimaan oikein. Painoarvojen osittaisderivaatan vektorisoidussa kaavassa ei puhuttuu dimensioista, mutta aktivointivektoria transponoimalla sain vektorit oikeaan asentoon.

Stokastisen gradienttimenetelmän osalta päädyin toteuttamaan sen generaattorina. SGD funktiolle antetaan Epoch arvo, ja jokaisen Epoch välissä yieldataan painot ja vakiot, jotta voidaan evaluoida verkon sen hetkinen tarkkuus.

Toteutin myös Python luokan tallentamaan (pickle) käytetyn kerroskonfiguraaton ja lasketut painot ja vakiot, koska opettaminen on hidasta. Näin voidaan palata eri versioihin helposti.

Tein useita kokeluija erilaisilla kerroskonfiguraatioilla. Sain sigmoid aktivoinnilla ja 784-30-16-10 verkolla 94.5% tarkkuuden. Notebook löytyy [täältä](src/neural_network.ipynb). Ensi viikolla toteutetaan ReLu ja Step funktio, sekä mahdollisesti jotain muita parannuksia mikäli materiaalisestani löytyy. 

Kysymys ohjaajalle: Kuinka tärkeää tässä projekissa yksikkötestaaminen on? Olen toteuttanut eriosiot niin, että niitä olisi helpompi testata eristyksissä neuroverkon muista osa-alueista. Kuitenkin nykyään ohjelmistoalalla on enemmän muodissa tehdä integraatiotestejä yksikkötestaamisen sijaan. Koska näiden algoritmien yksikkötestaaminen on hankalaa, parempi hyöty saadaan testaamalla opetuksen aikana validaatiosetillä ja opetuksen valmistumisen jälkeen testisetillä.
