import csv
import random
import math
from spellchecker import SpellChecker
import numpy as np
from backend.algorithms import datasets


'''Output resulting string after a pair of indices a and b are swapped in curr_str'''
def transition(ciphertext, curr_str):
    # generate two distinct random indices
    a = 0
    b = 0
    while (a == b):
        a = random.randint(0,27)
        b = random.randint(0,27)
    a, b = min(a,b), max(a,b)
    return curr_str[0:a] + curr_str[b] + curr_str[a+1:b] + curr_str[a] + curr_str[b+1:] 

'''Outputs the log of the likelihood p(ciphertext | key) along with decrypted message'''
def likelihood(ciphertext, key, datasets):
    # initial set_up and decoding of first letter
    prev_letter = datasets['alphabet_dict'][key[datasets['alphabet_dict'][ciphertext[0]]]]
    prob = np.log(float(datasets['letter_prob'][prev_letter]))
    plaintext = key[datasets['alphabet_dict'][ciphertext[0]]]

    # product of transition probabilities
    for i in range (1, len(ciphertext)):
        curr_letter = datasets['alphabet_dict'][key[datasets['alphabet_dict'][ciphertext[i]]]]
        prob += np.log(float(datasets['letter_trans'][curr_letter][prev_letter]))
        prev_letter = curr_letter
        plaintext += key[datasets['alphabet_dict'][ciphertext[i]]]
    return (prob, plaintext)

'''Outputs boolean whether or not to accept next_str, the next key, given 
log of curr_likelihood and next_likelihood'''
def accept(curr_likelihood, next_likelihood):
    threshhold = next_likelihood - curr_likelihood
    if (threshhold > 0):
        return True
    if (threshhold < -500):
        return False
    r = random.random()
    return (r <= min(np.exp(threshhold),1))

'''Outputs boolean whether or not to accept next_str, the next key, given 
log of curr_likelihood and next_likelihood'''
def accept_biased(curr_score, next_score, curr_likelihood, next_likelihood, it):
    # after sufficiently many iterations, bias transition if score is significantly higher
    if (next_score * 4 + next_likelihood > curr_score * 4 + curr_likelihood  and it > 2000):
        return True
    threshhold = next_likelihood - curr_likelihood
    if (threshhold > 0):
        return True
    if (threshhold < -500):
        return False
    r = random.random()
    return (r <= min(np.exp(threshhold),1))

'''Outputs the deciphered text given the ciphertext and key
    Note that since plaintext = key[alphabet_dict[ciphertext[i]]]
    if space maps to z, then ciphertext[i] = z, alphabet_dict[] = 25
    key[25] has to be space'''
def decipher(ciphertext, key, datasets):
    plaintext = ""
    for i in range(len(ciphertext)):
        plaintext += key[datasets['alphabet_dict'][ciphertext[i]]]
    return plaintext

'''finds the possible periods in a text'''
def find_possible_periods(ciphertext, space, char_freq):
    possible_periods = set(char_freq.copy())
    possible_periods.remove(ciphertext[0]) # rule: cannot begin with a period
    # remove all characters that follow a space
    for i in range(len(ciphertext)-1):
        if (ciphertext[i+1] != space and ciphertext[i] in possible_periods):
            possible_periods.remove(ciphertext[i])
    possible_periods = list(possible_periods)
    possible_periods.sort(key = lambda x: abs(char_freq[x]/len(ciphertext) - 0.00923425477857252))
    return possible_periods

'''finds all double letters in ciphertext and orders by frequency'''
def find_double_letters(ciphertext):
    # find all the double letters
    freqs = {}
    for i in range(len(ciphertext)-1):
        if (ciphertext[i] == ciphertext[i+1]):
            if (ciphertext[i] not in freqs.keys()):
                freqs[ciphertext[i]] = 1
            else:
                freqs[ciphertext[i]] += 1
    # order them by frequency
    double_letters = list(freqs.keys())
    double_letters.sort(key = lambda x: freqs[x], reverse = True)
    return double_letters

'''finds all single letter words in ciphertext and orders by frequency'''
def find_single_letters(ciphertext, space, char_freq):
    # char_freq not used (used in earlier version, might be useful)
    freqs = {}
    for i in range(len(ciphertext)-2):
        if (ciphertext[i] == space and ciphertext[i+2] == space):
            if (ciphertext[i+1] not in freqs.keys()):
                freqs[ciphertext[i+1]] = 1
            else:
                freqs[ciphertext[i+1]] += 1
    # exception is the last one
    if (ciphertext[len(ciphertext)-2] == space):
        if (ciphertext[len(ciphertext)-1] not in freqs.keys()):
            freqs[ciphertext[len(ciphertext)-1]] = 1
        else:
            freqs[ciphertext[len(ciphertext)-1]] += 1
    single_letters = list(freqs.keys())
    single_letters.sort(key = lambda x: freqs[x], reverse = True)
    return single_letters

''' initializes to a very good str based on frequencies and other statistics'''
def initial_str_best(ciphertext, datasets):
    # initialize string based on key
    char_freq_alphabet = datasets['alphabet'].copy()
    char_freq_alphabet.sort(key = lambda x: datasets['letter_prob'][datasets['alphabet_dict'][x]], reverse = True)
    char_freq = dict((char,0) for char in datasets['alphabet'])
    for char in ciphertext:
        char_freq[char] += 1
    char_freq_ciphertext = list(char_freq.keys())
    char_freq_ciphertext.sort(key = lambda x: char_freq[x], reverse = True)
    initial_key = [0] * 28
    for i in range(28):
        initial_key[datasets['alphabet_dict'][char_freq_ciphertext[i]]] = char_freq_alphabet[i]

    space = char_freq_ciphertext[0]
    possible_periods = find_possible_periods(ciphertext, space, char_freq)
    doubles = find_double_letters(ciphertext)
    singles = find_single_letters(ciphertext, space, char_freq)

    # swap to get double letters in place
    for i in range(len(doubles)-1,-1,-1):
        initial_key[initial_key.index(datasets['double_letters'][i])] = initial_key[datasets['alphabet_dict'][doubles[i]]]
        initial_key[datasets['alphabet_dict'][doubles[i]]] = datasets['double_letters'][i]

    # swap to get single letter words in place
    for i in range(min(len(singles)-1,2),-1,-1):
        initial_key[initial_key.index(datasets['single_letter_words'][i])] = initial_key[datasets['alphabet_dict'][singles[i]]]
        initial_key[datasets['alphabet_dict'][singles[i]]] = datasets['single_letter_words'][i]
    
    # swap to get period in place
    if (len(possible_periods) > 0):
        initial_key[initial_key.index('.')] = initial_key[datasets['alphabet_dict'][possible_periods[0]]]
        initial_key[datasets['alphabet_dict'][possible_periods[0]]] = '.'
    return "".join(initial_key)

''' initializes to a good str based on frequencies'''
def initial_str_by_freq(ciphertext, datasets):
    # compute char_freq
    char_freq_alphabet = datasets['alphabet'].copy()
    char_freq_alphabet.sort(key = lambda x: datasets['letter_prob'][datasets['alphabet_dict'][x]], reverse = True)
    char_freq = dict((char,0) for char in datasets['alphabet'])
    for char in ciphertext:
        char_freq[char] += 1
    char_freq_ciphertext = list(char_freq.keys())
    char_freq_ciphertext.sort(key = lambda x: char_freq[x], reverse = True)

    initial_key = [0] * 28
    for i in range(28):
        initial_key[datasets['alphabet_dict'][char_freq_ciphertext[i]]] = char_freq_alphabet[i]
    return "".join(initial_key)

'''Generates a random permutation of the alphabet'''
def random_str(datasets):
    # generate a random key to start at
    random_dict = {datasets["alphabet"][i] : random.random() for i in range(28)}    
    random_perm = list(random_dict.keys())
    random_perm.sort(key = lambda x: random_dict[x])
    curr_str = "".join(random_perm)
    return curr_str

def score(plaintext, datasets):
    points = 0 # sum of the number of characters in the total number of words correct
    words = plaintext.split()
    for w in words:
        if (w in datasets["english_dict"] or "." in w and w[:-1] in datasets["english_dict"]):
            points += len(w)
    return 2 * points

'''Outputs MAP of ciphertext via MCMC with no breakpoints'''
def decode_1_smartest(ciphertext, datasets):
    random.seed(314)
    it = 10000
    if len(ciphertext) <= 350:
        it = 11000
    elif len(ciphertext) <= 750:
        it = 5000
    elif len(ciphertext) <= 1400:
        it = 4500
    else:
        it = 3000

    if len(ciphertext) == 0:
        return {"plaintext" : "", "loglikelihood" : 0}

    # initialize string    
    curr_str = initial_str_best(ciphertext, datasets)

    # statistics to keep track of
    curr_likelihood, plaintext = likelihood(ciphertext, curr_str, datasets) # likelihood of the current key
    curr_score = score(plaintext, datasets)
    best_key = curr_str
    best_likelihood = curr_likelihood + curr_score


    # iterate through MCMC
    for i in range(it):
        # generate next key and compute likelihood
        next_str = transition(ciphertext, curr_str)
        next_likelihood, plaintext = likelihood(ciphertext, next_str, datasets)
        next_score = score(plaintext, datasets)

        # check if proposal passes
        if (accept_biased(curr_score, next_score, curr_likelihood, next_likelihood, i)):
            curr_str = next_str
            curr_likelihood = next_likelihood
            curr_score = next_score
            if (curr_likelihood + curr_score > best_likelihood):
                best_key = curr_str
                best_likelihood = curr_likelihood + curr_score
    return {"plaintext": decipher(ciphertext, best_key, datasets), "loglikelihood": best_likelihood}

'''Outputs MAP of ciphertext via MCMC with no breakpoints'''
def decode_1_smarter(ciphertext, datasets):
    random.seed(271)
    it = 20000
    if len(ciphertext) <= 350:
        it = 22000
    elif len(ciphertext) <= 750:
        it = 15000
    elif len(ciphertext) <= 1400:
        it = 10000
    else:
        it = 7000

    if len(ciphertext) == 0:
        return {"plaintext" : "", "loglikelihood" : 0}

    # initialize string    
    curr_str = initial_str_by_freq(ciphertext, datasets)

    # statistics to keep track of
    curr_likelihood, plaintext = likelihood(ciphertext, curr_str, datasets) # likelihood of the current key
    curr_score = score(plaintext, datasets)
    best_key = curr_str
    best_likelihood = curr_likelihood + curr_score

    # iterate through MCMC
    for i in range(it):
        # generate next key and compute likelihood
        next_str = transition(ciphertext, curr_str)
        next_likelihood, plaintext = likelihood(ciphertext, next_str, datasets)
        next_score = score(plaintext, datasets)

        # check if proposal passes
        if (accept(curr_likelihood, next_likelihood)):
            curr_str = next_str
            curr_likelihood = next_likelihood
            curr_score = next_score
            if (curr_likelihood + curr_score > best_likelihood):
                best_key = curr_str
                best_likelihood = curr_likelihood + curr_score

    return {"plaintext": decipher(ciphertext, best_key, datasets), "loglikelihood": best_likelihood}

'''Outputs MAP of ciphertext via MCMC with no breakpoints'''
def decode_1_dumbest(ciphertext, datasets):
    random.seed(577)
    it = 20000
    if len(ciphertext) <= 350:
        it = 15000
    elif len(ciphertext) <= 750:
        it = 10000
    elif len(ciphertext) <= 1400:
        it = 7000
    else:
        it = 4000

    if len(ciphertext) == 0:
        return {"plaintext" : "", "loglikelihood" : 0}

    # initialize string    
    curr_str = random_str(datasets)

    # statistics to keep track of
    curr_likelihood, plaintext = likelihood(ciphertext, curr_str, datasets) # likelihood of the current key
    curr_score = score(plaintext, datasets)
    best_key = curr_str
    best_likelihood = curr_likelihood + curr_score

    # iterate through MCMC
    for i in range(it):
        # generate next key and compute likelihood
        next_str = transition(ciphertext, curr_str)
        next_likelihood, plaintext = likelihood(ciphertext, next_str, datasets)
        next_score = score(plaintext, datasets)

        # check if proposal passes
        if (accept(curr_likelihood, next_likelihood)):
            curr_str = next_str
            curr_likelihood = next_likelihood
            curr_score = next_score
            if (curr_likelihood + curr_score > best_likelihood):
                best_key = curr_str
                best_likelihood = curr_likelihood + curr_score

    return {"plaintext": decipher(ciphertext, best_key, datasets), "loglikelihood": best_likelihood}

'''Main decode function for part_1, calls decode dumbest, smarter, and smartest'''
def decode_part_1(ciphertext, datasets):
    smartest = decode_1_smartest(ciphertext, datasets)
    smarter = decode_1_smarter(ciphertext, datasets)
    dumbest = decode_1_dumbest(ciphertext, datasets)
    return max([smartest,smarter,dumbest], key = lambda x: x["loglikelihood"])

'''part_1 decryption used in part 2'''
def decode_part_2_rough(ciphertext, datasets):
    it = min(1800000//len(ciphertext),9500)
    if len(ciphertext) == 0:
        return {"plaintext" : "", "loglikelihood" : 0}

    # initialize string    
    curr_str = random_str(datasets)

    # statistics to keep track of
    curr_likelihood, _ = likelihood(ciphertext, curr_str, datasets) # likelihood of the current key
    best_likelihood = curr_likelihood
    best_key = curr_str

    # iterate through MCMC
    for i in range(it):
        # generate next key and compute likelihood
        next_str = transition(ciphertext, curr_str)
        next_likelihood, _ = likelihood(ciphertext, next_str, datasets)

        # check if proposal passes
        if (accept(curr_likelihood, next_likelihood)):
            curr_str = next_str
            curr_likelihood = next_likelihood
            if (curr_likelihood > best_likelihood):
                best_likelihood = curr_likelihood
                best_key = curr_str

    return {"plaintext": decipher(ciphertext, best_key, datasets), "loglikelihood": best_likelihood}

'''Outputs modified notion of divergence based on KL divergence
    we use the sum of two KL divergences to force the notion of symmetry
    and we form d1_adj and d2_adj to avoid div by 0 and as some sort of prior'''
def divergence(index, ciphertext_len, frequencies_forward, frequencies_backward, alphabet):
    freq_1 = frequencies_forward[index - 1] # first index letters
    freq_2 = frequencies_backward[ciphertext_len - index - 1] # last ciphertext_len - index letters
    total = 0
    alpha = 1/2

    for letter in alphabet:
        d1_adj = (freq_1[letter] + alpha)/(index + 28 * alpha)
        d2_adj = (freq_2[letter] + alpha)/(ciphertext_len - index + 28*alpha)
        total += d1_adj * np.log(d1_adj/d2_adj) + d2_adj * np.log(d2_adj/d1_adj)

    return(total)

'''Outputs a list of possible breakpoints sorted in decreasing order of divergence'''
def find_bp(ciphertext, alphabet):
    frequencies_forward = []
    frequencies_backward = []
    divergences = {}

    # compute char_freq moving forwards: index i has the first i+1 letters
    char_freq = dict((char,0) for char in alphabet)
    for char in ciphertext:
        char_freq[char] += 1
        frequencies_forward.append(char_freq.copy())
    
    # compute char_freq moving backwards index i has the last i+1 letters
    char_freq = dict((char,0) for char in alphabet)
    for i in range(len(ciphertext)-1,-1,-1):
        char_freq[ciphertext[i]] += 1
        frequencies_backward.append(char_freq.copy())

    # here j is the number of letters in the first part
    for j in range(1,len(ciphertext)):
        divergences[j] = divergence(j, len(ciphertext), frequencies_forward, frequencies_backward, alphabet)    
    
    index_list = list(divergences.keys())
    index_list.sort(key = lambda x: divergences[x], reverse = True) 
    return index_list

'''Outputs MAP estimate when there is a breakpoint'''
def decode_part_2(ciphertext, datasets):
    index_list = find_bp(ciphertext, datasets["alphabet"])
    bp_1 = index_list[0]
    bp_2 = bp_1
    best_plaintext = ""
    best_likelihood = -1e20

    # the first index in find_bp automatically qualifies for final comparisons
    for i in range(1,15):
        pre_bp = ciphertext[:index_list[i]]
        post_bp = ciphertext[index_list[i]:]
        pre_bp_decoded = decode_part_2_rough(pre_bp, datasets)
        post_bp_decoded = decode_part_2_rough(post_bp, datasets)
        post_decoded_tog = pre_bp_decoded["plaintext"] + post_bp_decoded["plaintext"]
        likelihood = pre_bp_decoded["loglikelihood"] + post_bp_decoded["loglikelihood"]
        if (len(pre_bp_decoded["plaintext"]) * len(post_bp_decoded["plaintext"]) > 0):
            likelihood += datasets["letter_trans"][datasets['alphabet_dict'][post_bp_decoded["plaintext"][0]]][datasets['alphabet_dict'][pre_bp_decoded["plaintext"][-1]]]
        if (likelihood > best_likelihood):
            best_plaintext = post_decoded_tog
            best_likelihood = likelihood
            bp_2 = index_list[i]

    # consider edge case where breakpoint is on very edge so there is no breakpoint
    edge_case = decode_part_2_rough(ciphertext, datasets)
    if (edge_case["loglikelihood"] > best_likelihood):
        best_plaintext = edge_case["plaintext"]
        best_likelihood = edge_case["loglikelihood"]
        bp_2 = 0
    
    plaintext_dict = {best_plaintext : best_likelihood + score(best_plaintext, datasets)}

    pre_bp = ciphertext[:bp_1]
    post_bp = ciphertext[bp_1:]
    pre_bp_decoded = decode_1_smartest(pre_bp, datasets)
    post_bp_decoded = decode_1_smartest(post_bp, datasets)
    post_decoded_tog = pre_bp_decoded["plaintext"] + post_bp_decoded["plaintext"]
    likelihood = pre_bp_decoded["loglikelihood"] + post_bp_decoded["loglikelihood"]
    if (len(pre_bp_decoded["plaintext"]) * len(post_bp_decoded["plaintext"]) > 0):
        likelihood += datasets["letter_trans"][datasets['alphabet_dict'][post_bp_decoded["plaintext"][0]]][datasets['alphabet_dict'][pre_bp_decoded["plaintext"][-1]]]
    plaintext_dict[post_decoded_tog] = likelihood

    pre_bp = ciphertext[:bp_1]
    post_bp = ciphertext[bp_1:]
    pre_bp_decoded = decode_1_dumbest(pre_bp, datasets)
    post_bp_decoded = decode_1_dumbest(post_bp, datasets)
    post_decoded_tog = pre_bp_decoded["plaintext"] + post_bp_decoded["plaintext"]
    likelihood = pre_bp_decoded["loglikelihood"] + post_bp_decoded["loglikelihood"]
    if (len(pre_bp_decoded["plaintext"]) * len(post_bp_decoded["plaintext"]) > 0):
        likelihood += datasets["letter_trans"][datasets['alphabet_dict'][post_bp_decoded["plaintext"][0]]][datasets['alphabet_dict'][pre_bp_decoded["plaintext"][-1]]]
    plaintext_dict[post_decoded_tog] = likelihood

    pre_bp = ciphertext[:bp_2]
    post_bp = ciphertext[bp_2:]
    pre_bp_decoded = decode_1_smartest(pre_bp, datasets)
    post_bp_decoded = decode_1_smartest(post_bp, datasets)
    post_decoded_tog = pre_bp_decoded["plaintext"] + post_bp_decoded["plaintext"]
    likelihood = pre_bp_decoded["loglikelihood"] + post_bp_decoded["loglikelihood"]
    if (len(pre_bp_decoded["plaintext"]) * len(post_bp_decoded["plaintext"]) > 0):
        likelihood += datasets["letter_trans"][datasets['alphabet_dict'][post_bp_decoded["plaintext"][0]]][datasets['alphabet_dict'][pre_bp_decoded["plaintext"][-1]]]
    plaintext_dict[post_decoded_tog] = likelihood

    plaintext = max(plaintext_dict.keys(), key = lambda x: plaintext_dict[x])
    return plaintext

'''Cleans up final result using a spellchecker'''
def cleanup(plaintext, datasets):
    # only correct if one letter is off
    spell = SpellChecker(distance = 1)
    words = plaintext.split()
    for i in range(len(words)):
        # consider casework on periods and guarantee words are same length as before
        if "." not in words[i] and words[i] not in datasets["english_dict"]:
            correction = spell.correction(words[i])
            if (len(correction) == len(words[i])):
                words[i] = correction
        elif words[i].find(".") == len(words[i]) - 1:
            word_trunk = words[i][:-1]
            if "." not in word_trunk and word_trunk not in datasets["english_dict"]:
                correction = spell.correction(word_trunk)
                if (len(correction) == len(word_trunk)):
                    words[i] = correction + "."
    return " ".join(words)

'''Evaluates accuracy given two piece of plaintext'''
def accuracy(deciphered_message):
    fin = open("./plaintext.txt", "r")
    plaintext = fin.read()
    fin.close()
    correct_char = 0
    for i in range(len(deciphered_message)):
        if (plaintext[i] == deciphered_message[i]):
            correct_char += 1
    return correct_char / len(deciphered_message)

'''main decode function for project, broken by case on has_breakpoint'''
def decode(ciphertext, has_breakpoint):
    plaintext = ""
    if (has_breakpoint):
        plaintext = decode_part_2(ciphertext, datasets)
        # print("accuracy is " + str(accuracy(plaintext)))
        plaintext = cleanup(plaintext, datasets)
    else:
        plaintext = decode_part_1(ciphertext, datasets)["plaintext"]
    # print("final accuracy is " + str(accuracy(plaintext)))
    return plaintext

'''web version of decode'''
def decode_web_version(ciphertext):
    random.seed(314)
    original_ciphertext = ciphertext
    ciphertext = ciphertext[:600]
    it = 15000

    if len(ciphertext) == 0:
        return {"plaintext" : "", "loglikelihood" : 0}

    # initialize string    
    curr_str = initial_str_best(ciphertext, datasets)

    # statistics to keep track of
    curr_likelihood, plaintext = likelihood(ciphertext, curr_str, datasets) # likelihood of the current key
    curr_score = score(plaintext, datasets)
    best_key = curr_str
    best_likelihood = curr_likelihood + curr_score


    # iterate through MCMC
    for i in range(it):
        # generate next key and compute likelihood
        next_str = transition(ciphertext, curr_str)
        next_likelihood, plaintext = likelihood(ciphertext, next_str, datasets)
        next_score = score(plaintext, datasets)

        # check if proposal passes
        if (accept_biased(curr_score, next_score, curr_likelihood, next_likelihood, i)):
            curr_str = next_str
            curr_likelihood = next_likelihood
            curr_score = next_score
            if (curr_likelihood + curr_score > best_likelihood):
                best_key = curr_str
                best_likelihood = curr_likelihood + curr_score
    return decipher(original_ciphertext, best_key, datasets)


''' main function of module '''
def main():
    # when testing with encode.py use ciphertext.txt and plaintext.txt as inputs
    # decoded text at end written in decoded_text.txt
    random.seed(162)
    fin = open("./ciphertext.txt", "r")
    ciphertext = fin.read()
    fin.close()
    plaintext = decode(ciphertext, False)
    fout = open("./decoded_text.txt", "w")
    fout.write(plaintext)
    fout.close()

if __name__ == "__main__":
    main()
