#!/bin/bash

# Define the array with some values
# lambdas=(3 4 7)
# # lambdas=(1 10)
# # sigmas=(0.08 0.12)


# # Iterate over the arrays with indices
# for x in "${!lambdas[@]}"; do
#     for y in "${!sigmas[@]}"; do
#         lambda="${lambdas[$x]}"
#         sigma="${sigmas[$y]}"
#         echo "Lambda Index: $x, Value: $lambda, Sigma Index: $y, Value: $sigma"
#         # Run Python script with the current index and value
#         python test.py  --config-name=conf_piano2.yaml tester.checkpoint="./experiments/ckpt.pt" id="BABE2" tester.modes=["dpir"] dpir.test.x=$x dpir.test.y=$y  dpir.lambda_=$lambda dpir.sigma_noise=$sigma
#     done
# done


lambdas=(5)
snr_DB=(1.0 3.0 5.0 7.0 10.0 15.0 20.0)


# Iterate over the arrays with indices
for x in "${!lambdas[@]}"; do
    for y in "${!snr_DB[@]}"; do
        lambda="${lambdas[$x]}"
        snr_db="${snr_DB[$y]}"
        echo "Lambda Index: $x, Value: $lambda, Sigma Index: $y, Value: $snr_db"
        # Run Python script with the current index and value
        id="declip_${snr_db}_lam_${lambda}"
        python test.py  --config-name=conf_piano2.yaml tester.checkpoint="./experiments/ckpt.pt" id=$id tester.modes=["dpir"] dpir.lambda_=$lambda dpir.snr_dB=1.0 dpir.declipping.sdr=$snr_db
    done
done