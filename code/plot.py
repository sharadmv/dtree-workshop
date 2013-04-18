import pylab as plt
"""
trerr = [0.3973898858075041, 0.2013050570962479, 0.13278955954323002, 0.1252854812398042, 0.08972267536704726, 0.07471451876019575, 0.06199021207177813, 0.053181076672104366, 0.04469820554649262, 0.037520391517128826, 0.03262642740619903, 0.028711256117455175, 0.024796084828711207, 0.02022838499184343, 0.017292006525285508, 0.01272430668841762, 0.009135399673735778, 0.008482871125611746, 0.0055464926590538255, 0.003915171288743857, 0.0029363784665579207, 0.002283849918433889, 0.0006525285481240317, 0.00032626427406201586, 0.00032626427406201586, 0.00032626427406201586]
teerr = [0.38736979166666663, 0.21614583333333337, 0.14127604166666663, 0.13411458333333337, 0.099609375, 0.08658854166666663, 0.08854166666666663, 0.08138020833333337, 0.083984375, 0.08463541666666663, 0.08919270833333337, 0.083984375, 0.083984375, 0.09049479166666663, 0.08723958333333337, 0.09244791666666663, 0.091796875, 0.09440104166666663, 0.09309895833333337, 0.09505208333333337, 0.09505208333333337, 0.09700520833333337, 0.095703125, 0.095703125, 0.095703125, 0.095703125]
training = plt.plot(range(0,26), trerr, label="Training")
test = plt.plot(range(0, 26), teerr, label="Test")
plt.ylabel("Error")
plt.xlabel("Depth")
plt.legend()
plt.show()
"""
"""
trerr = [0.15138662316476348, 0.1468189233278956, 0.09918433931484505, 0.09200652528548126, 0.06982055464926595, 0.07504078303425776, 0.06851549755301789, 0.07699836867862964, 0.06590538336052199, 0.05970636215334424, 0.05448613376835232, 0.05742251223491024, 0.05350734094616638, 0.05676998368678632, 0.060032626427406144, 0.05448613376835232, 0.05448613376835232, 0.053181076672104366, 0.05350734094616638, 0.052528548123980445, 0.054159869494290414, 0.04991843393148454, 0.04828711256117457, 0.049592169657422525, 0.04828711256117457, 0.051876019575856414, 0.051876019575856414, 0.0538336052202284, 0.05089722675367048, 0.04828711256117457, 0.04861337683523659, 0.04991843393148454, 0.053181076672104366, 0.05089722675367048, 0.05089722675367048, 0.05742251223491024, 0.052528548123980445, 0.04567699836867867, 0.055464926590538366, 0.051876019575856414, 0.051876019575856414, 0.045024469820554636, 0.04861337683523659, 0.047308319738988636, 0.047308319738988636, 0.04828711256117457, 0.05057096247960846, 0.04632952691680259, 0.047960848287112556, 0.047308319738988636, 0.047960848287112556, 0.04600326264274057, 0.05057096247960846, 0.04828711256117457, 0.050244698205546445, 0.04861337683523659, 0.04861337683523659, 0.050244698205546445, 0.05122349102773249, 0.047960848287112556, 0.05089722675367048, 0.04828711256117457, 0.05350734094616638, 0.04828711256117457, 0.04926590538336051, 0.04893964110929849, 0.04926590538336051, 0.04763458401305054, 0.046655791190864604, 0.047960848287112556, 0.04371941272430668, 0.047308319738988636, 0.04861337683523659, 0.04828711256117457, 0.04371941272430668, 0.047308319738988636, 0.04535073409461665, 0.046655791190864604, 0.04600326264274057, 0.047960848287112556, 0.047960848287112556, 0.051876019575856414, 0.04535073409461665, 0.04828711256117457, 0.04926590538336051, 0.050244698205546445, 0.04632952691680259, 0.04763458401305054, 0.04861337683523659, 0.047960848287112556, 0.05057096247960846, 0.046655791190864604, 0.04926590538336051, 0.049592169657422525, 0.04567699836867867, 0.046655791190864604, 0.047308319738988636, 0.047960848287112556, 0.04763458401305054, 0.047308319738988636]
teerr = [0.16731770833333337, 0.16145833333333337, 0.11263020833333337, 0.11328125, 0.09765625, 0.103515625, 0.07942708333333337, 0.09895833333333337, 0.08072916666666663, 0.07552083333333337, 0.078125, 0.07291666666666663, 0.06380208333333337, 0.0859375, 0.07942708333333337, 0.080078125, 0.07096354166666663, 0.06380208333333337, 0.06575520833333337, 0.06315104166666663, 0.076171875, 0.06510416666666663, 0.072265625, 0.06510416666666663, 0.07161458333333337, 0.06119791666666663, 0.07096354166666663, 0.068359375, 0.068359375, 0.06380208333333337, 0.064453125, 0.076171875, 0.06510416666666663, 0.06510416666666663, 0.068359375, 0.068359375, 0.068359375, 0.06901041666666663, 0.06575520833333337, 0.068359375, 0.06575520833333337, 0.068359375, 0.05859375, 0.064453125, 0.06510416666666663, 0.0703125, 0.07096354166666663, 0.06575520833333337, 0.06705729166666663, 0.06380208333333337, 0.068359375, 0.06119791666666663, 0.06380208333333337, 0.06184895833333337, 0.05989583333333337, 0.06705729166666663, 0.06705729166666663, 0.06380208333333337, 0.06380208333333337, 0.064453125, 0.0625, 0.064453125, 0.06705729166666663, 0.0625, 0.06184895833333337, 0.06575520833333337, 0.060546875, 0.064453125, 0.06380208333333337, 0.06315104166666663, 0.06119791666666663, 0.06510416666666663, 0.06705729166666663, 0.06640625, 0.064453125, 0.068359375, 0.06380208333333337, 0.05924479166666663, 0.05859375, 0.068359375, 0.064453125, 0.05729166666666663, 0.0625, 0.07096354166666663, 0.06510416666666663, 0.06380208333333337, 0.064453125, 0.05794270833333337, 0.06119791666666663, 0.068359375, 0.06184895833333337, 0.0625, 0.06119791666666663, 0.064453125, 0.05924479166666663, 0.06184895833333337, 0.06705729166666663, 0.0625, 0.06315104166666663, 0.06184895833333337]
training = plt.plot(range(0,100), trerr, label="Training")
test = plt.plot(range(0, 100), teerr, label="Test")
"""
trerr = [0.08254486133768357, 0.05220228384991843, 0.05057096247960846, 0.050244698205546445, 0.04600326264274057, 0.04763458401305054, 0.047960848287112556, 0.047308319738988636, 0.04698205546492662, 0.04698205546492662]
teerr = [0.08854166666666663, 0.06640625, 0.06575520833333337, 0.05794270833333337, 0.06315104166666663, 0.05859375, 0.0625, 0.06510416666666663, 0.0625, 0.06315104166666663]
training = plt.plot(range(1,11), trerr, label="Training")
test = plt.plot(range(1, 11), teerr, label="Test")
plt.ylabel("Error")
plt.xlabel("Number of Features")
plt.legend()
plt.show()
