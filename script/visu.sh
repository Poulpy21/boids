gnuplot -e "do for [i=1:$1] {splot 'boids.xyz' using 2:3:4 every :::i::i; pause .1}"
