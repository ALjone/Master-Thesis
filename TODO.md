Prio:
    Invertere Rosenbrock?
    Teste uten GP som input?
    Fiks config, så man bare trenger å forandre ting i en av de
    Lage forskjellige klasser av random functions som arver fra en hovedklasse
    Implementere Euclidean distance transform
    Teste på RF med andre params
    Input channel arange delt på max ref Sigurd og Herman.
    Add dimension to state with distance to closest tested points
    Pretraine på EI
    Logge, underveis, distansen fra maks EI til maks sannsynlighetstetthet
    Hyperparameter søk på epsilon i EI
    

Skrive om reward sånn at den gir mening

Normaliserer nå i random function, men heller gjøre det med sampled points.

Se mer på random functions, tune de bra

Skrive en visualiserings-ting, som viser både action, distanse fra forrige/alle, og critic

Bruke det faktum av CleanRL lagrer alle actions, til å visualisere hvor de skjer i snitt med noe histogram-greier