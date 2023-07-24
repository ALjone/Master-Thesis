Prio:
    Performance metric, max of peak isn't necessarily good (if there is a very deep minima, everything else will be squashed towards close to max)
    Fiks config, så man bare trenger å forandre ting i en av de
    Lage forskjellige klasser av random functions som arver fra en hovedklasse
    Teste på RF med andre params
    Hyperparameter søk på epsilon i EI
    

Skrive om reward sånn at den gir mening

Normaliserer nå i random function, men heller gjøre det med sampled points.

Se mer på random functions, tune de bra

Skrive en visualiserings-ting, som viser både action, distanse fra forrige/alle, og critic

Bruke det faktum av CleanRL lagrer alle actions, til å visualisere hvor de skjer i snitt med noe histogram-greier





Teste grid + zoom in grid
Ha stor forskjell i tiden man har til rådighet per env, for å lære forskjellige ting

Eksperiment: 
1. Trene og teste på en type funksjon, med en type tid
2. Trene og teste på mange typer funksjoner, med en type tid
3. Trene og teste på mange typer funksjoner med mange typer tider
4. Teste 2 og 3 i høyere dims?
5. Trene og teste en model ala 1-3 uten tilgang til tid. 
6. Faller den tilbake på vanlig performance hvis det ikke er noe cost-awareness? IKKE SIKKERT DETTE TRENGS
7. Hva som skjer når man har _mye_ tid?

Baseline:
Prøve på den divide by coost cooling greien?
Holder med no-time, for MetaBO viser at den funker bedre enn f.eks TO?
Trene faktisk uten tid, der alt er samme kost, for å sjekke skikkelig hva det har å si?

TODO:
Make EI/UCB/PI agent
Add Noise?