$N = 20; 
$k_max = 481;
$j_max = $k_max/$N;

$i_biggest = $k_max -1;

for $j (0..$j_max){
    print "round $j/$j_max: "; system "date";
    $i_max = ($j+1)*$N -1;
    if ($i_max > $i_biggest) {
        $i_max = $i_biggest;
    }
    for $i ($j*$N..$i_max) {
            print "fork $i: "; system "date";
            my $pid = fork();
            if (not $pid) {
                    system("python -m EmilyWork.WFGenerator 80 $i Hg_k_$i_nmax_1200_Nr_500  500 1e-15");
                    exit();
            }
    }
    for $i (0..$N-1) {
            wait();
            print "waited $i/$N: "; system "date";
    }
}