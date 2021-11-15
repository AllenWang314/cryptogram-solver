# cryptogram-solver
A MCMC-based algorithm that solves substitution cryptograms. A demo of part 1 of the project is deployed [here](https://cryptogram.allenwang314.com).

## Background
The project was the final assignment in MIT's 6.437 Inference and Information. In the first part of the project, students created an MCMC algorithm that decrypted a substitution cryptogram where 28 characters (the alphabet, space char, and period) were used in the key. In the second part, the goal was to improve the overall accuracy of the first part, but also allow the algorithm to decrypt messages with a *breakpoint*, a point in the text where the key was changed.

Information about the project, my solution, and my thought process can be found in `final_report/r_d_report.pdf`.

## Relevant Files

- `requirements.txt` contains the requirements to be downloaded
- `old_code/decode_part_1.py` contains the main functions used for part I
- `old_code/decode_part_1_better.py` contains improved versions of functions in `decode_part_1.py`
- `decode.py` contains the final version of the decode function that supports breakpoints
- `encode.py` is a script that encrypts some message into two files
- `data` is a folder with some sample texts and useful datasets
- `test.py` is a function that tests the decode function in `decode.py` using the ciphertext and plaintext in the data folder

## Results

- The performance without breakpoints turned out to be in the high 90% range
- The preformance with breakpoints was in the high 90% range as well (though during testing, it timed out on a few inputs)
- Project was overall highly enjoyable :)
