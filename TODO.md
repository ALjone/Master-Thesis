Prio:
    Skrive om baseline gpy til å bruke batching, så det går fortere. 
    Gjøre skikkelig hyperparameter søk med GPY (Reso = 20?)
    Logge std output
    

Skrive om reward sånn at den gir mening

Normaliserer nå i random function, men heller gjøre det med sampled points.

Se mer på random functions, tune de bra

Pix2Pix! (Men kvisleis?)

Legge til en måte å sette noise i likelihood på.

Skrive en visualiserings-ting, som viser både action, distanse fra forrige/alle, og critic

Lage en baseline med EI, expected improvement, som viser hvor bra GP er

Bruke det faktum av CleanRL lagrer alle actions, til å visualisere hvor de skjer i snitt med noe histogram-greier

Fikse jitter. Resette? "NumericalWarning: A not p.d., added jitter of 1.0e-06 to the diagonal"

Sammenligne GP med random med tid, og uten tid

Legge til "Use GP" i Batched env, samt device