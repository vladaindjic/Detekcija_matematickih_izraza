# Detekcija_matematickih_izraza
Detekcija i izracunavanje matematickih izraza prikazanih na slici.

Implementirane su jednostavna konvolucijska i jednostavna neuralna mreza.
Na pripremljenom skupu slika procenat uspesnosti pri treniranju CNN je 100%,
a NN 96,88%. Obe imaju istu uspesnost na izrazima datim za izracunavanje i iznosi
100%. S obzirom da se NN daleko brze trenira od CNN, za ovaj problem je
pogodnija upotreba NN.

Pokretanje aplikacije:
  1. Klonirati git repozitorijum ili ga kopirati u .zip formatu
  2. izbrisati outputImages direktorijum kako bi se obavio proces pripreme dataseta
  3. izbrisati simpleCNNModel.hdf5 kako bi se obavio proces treniranja CNN
  4. izbrisati simpleNNModel.hdf5 kako bi se obavio proces treniranja NN
  5. pokrenuti aplikaciju komandom "$ python expressionCalculator.py"
