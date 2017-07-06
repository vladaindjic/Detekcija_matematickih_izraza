# Detekcija_matematickih_izraza
Detekcija i izracunavanje matematickih izraza prikazanih na slici.

Implementirane su jednostavna konvolucijska i jednostavna neuralna mreza.
Na pripremljenom skupu slika (dataset) procenat uspesnosti pri treniranju obe mreze iznosi 100%. Obe imaju istu uspesnost u prepoznavanju izraza datih na 50 slika u folderu "expressions", koja iznosi
94-96%. S obzirom da se NN daleko brze trenira od CNN, za ovaj problem je
pogodnija upotreba NN.

Pokretanje aplikacije:
  1. Klonirati git repozitorijum ili ga kopirati u .zip formatu
  2. Ukoliko se zeli koristiti pripremljen dataset, kao i istrenirane NN i CNN pokrenuti aplikaciju komandom "$ python expressionCalculator.py"
  3. Za pravljenje dataseta i treniranje NN i CNN pokrenuti aplikaciju komandom "$ python expressionCalculator.py clear"
