




# 有考虑SGD
left_right_datas = [{
  "testdic": {
    "prec": 0.8763532763532763,
    "rec": 0.7697697697697697,
    "f1": 0.8196109778843592,
    "bacc": 0.8731488113316128
  },
  "devdic": {
    "prec": 0.6579406631762653,
    "rec": 0.556047197640118,
    "f1": 0.6027178257394085,
    "bacc": 0.7556338341410551
  },
  "losses": [
    6705.435110092134,
    4177.033804090373
  ],
  "desc": "左右池化，跑五次，和2vs2对比, flrate=0"
},
{
  "testdic": {
    "prec": 0.9642857142857143,
    "rec": 0.7162162162162162,
    "f1": 0.8219414129810453,
    "bacc": 0.8552416938301199
  },
  "devdic": {
    "prec": 0.8855218855218855,
    "rec": 0.387905604719764,
    "f1": 0.5394871794871795,
    "bacc": 0.6900688635890344
  },
  "losses": [
    6662.573991818877,
    4142.448828316119
  ],
  "desc": "左右池化，跑五次，和2vs2对比, flrate=0"
},
{
  "testdic": {
    "prec": 0.7957712638154734,
    "rec": 0.8288288288288288,
    "f1": 0.8119637165972053,
    "bacc": 0.8914290169022456
  },
  "devdic": {
    "prec": 0.49631190727081137,
    "rec": 0.6946902654867256,
    "f1": 0.5789797172710509,
    "bacc": 0.7927415229649758
  },
  "losses": [
    6733.502514963679,
    4400.167658648628
  ],
  "desc": "左右池化，跑五次，和2vs2对比, flrate=0"
},
{
  "testdic": {
    "prec": 0.9433719433719434,
    "rec": 0.7337337337337337,
    "f1": 0.8254504504504505,
    "bacc": 0.8621075374996414
  },
  "devdic": {
    "prec": 0.7181069958847737,
    "rec": 0.5147492625368731,
    "f1": 0.5996563573883161,
    "bacc": 0.7417246426917858
  },
  "losses": [
    6580.3796633508755,
    4178.9365173777915
  ],
  "desc": "左右池化，跑五次，和2vs2对比, flrate=0"
},
{
  "testdic": {
    "prec": 0.8938775510204081,
    "rec": 0.7672672672672672,
    "f1": 0.8257473740910315,
    "bacc": 0.8737904751695991
  },
  "devdic": {
    "prec": 0.6132231404958678,
    "rec": 0.5471976401179941,
    "f1": 0.578332034294622,
    "bacc": 0.7468681826361047
  },
  "losses": [
    6707.564587983623,
    4333.448989800207
  ],
  "desc": "左右池化，跑五次，和2vs2对比, flrate=0"
},
{
  "testdic": {
    "prec": 0.8745724059293044,
    "rec": 0.7677677677677678,
    "f1": 0.8176972281449894,
    "bacc": 0.8719855604658201
  },
  "devdic": {
    "prec": 0.6765249537892791,
    "rec": 0.5398230088495575,
    "f1": 0.6004922067268254,
    "bacc": 0.7499206431042396
  },
  "losses": [
    6722.739509095205,
    4368.866117107522
  ],
  "desc": "左右池化，跑五次，和2vs2对比, flrate=0"
},
{
  "testdic": {
    "prec": 0.9778085991678225,
    "rec": 0.7057057057057057,
    "f1": 0.8197674418604651,
    "bacc": 0.8511221876284072
  },
  "devdic": {
    "prec": 0.8413173652694611,
    "rec": 0.4144542772861357,
    "f1": 0.5553359683794465,
    "bacc": 0.701172763500276
  },
  "losses": [
    6640.006185412523,
    4315.684648919341
  ],
  "desc": "左右池化，跑五次，和2vs2对比, flrate=0"
},
]

# 没有考虑SGD
single_datas = [
{
  "testdic": {
    "prec": 0.862390350877193,
    "rec": 0.7872872872872873,
    "f1": 0.8231292517006802,
    "bacc": 0.8800687382893981
  },
  "devdic": {
    "prec": 0.6262458471760798,
    "rec": 0.556047197640118,
    "f1": 0.5890624999999999,
    "bacc": 0.7523210628365087
  },
  "losses": [
    6972.804297693889,
    5114.6519891172065
  ],
  "desc": "1:1 Double_Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9790356394129979,
    "rec": 0.7012012012012012,
    "f1": 0.8171478565179353,
    "bacc": 0.8489781019526828
  },
  "devdic": {
    "prec": 0.8474025974025974,
    "rec": 0.38495575221238937,
    "f1": 0.5294117647058824,
    "bacc": 0.6871089019229641
  },
  "losses": [
    6986.633322476351,
    5033.6920641800825
  ],
  "desc": "1:1 Double_Sentence_CLS, flrate=0"
},
]

# 有考虑SGD
single_sgd = [
{
  "testdic": {
    "prec": 0.9045871559633027,
    "rec": 0.7402402402402403,
    "f1": 0.8142031379025599,
    "bacc": 0.8616831271509475
  },
  "devdic": {
    "prec": 0.682261208576998,
    "rec": 0.5162241887905604,
    "f1": 0.5877413937867338,
    "bacc": 0.7394920349938637
  },
  "losses": [
    7066.007806696172,
    5521.954140776099
  ],
  "desc": "1:1 Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9479166666666666,
    "rec": 0.7287287287287287,
    "f1": 0.8239954725523486,
    "bacc": 0.8600377013032503
  },
  "devdic": {
    "prec": 0.8010335917312662,
    "rec": 0.45722713864306785,
    "f1": 0.5821596244131456,
    "bacc": 0.7198175903404966
  },
  "losses": [
    7049.445285115915,
    5236.0255624653655
  ],
  "desc": "1:1 Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.8833619210977701,
    "rec": 0.7732732732732732,
    "f1": 0.8246597277822257,
    "bacc": 0.8756036458307956
  },
  "devdic": {
    "prec": 0.6845360824742268,
    "rec": 0.4896755162241888,
    "f1": 0.5709372312983663,
    "bacc": 0.7273600336432802
  },
  "losses": [
    6896.2014626593445,
    4984.408144817877
  ],
  "desc": "1:1 Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.964405641370047,
    "rec": 0.7187187187187187,
    "f1": 0.8236306280470318,
    "bacc": 0.8564929450813712
  },
  "devdic": {
    "prec": 0.8021680216802168,
    "rec": 0.4365781710914454,
    "f1": 0.565425023877746,
    "bacc": 0.7099500405377264
  },
  "losses": [
    7061.878531100345,
    5337.268742520231
  ],
  "desc": "1:1 Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9573490813648294,
    "rec": 0.7302302302302303,
    "f1": 0.8285065303804656,
    "bacc": 0.8615997013779599
  },
  "devdic": {
    "prec": 0.836676217765043,
    "rec": 0.4306784660766962,
    "f1": 0.5686465433300877,
    "bacc": 0.7088279239225154
  },
  "losses": [
    7022.40968089184,
    5268.9091383012565
  ],
  "desc": "1:1 Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9511551155115512,
    "rec": 0.7212212212212212,
    "f1": 0.820381440364361,
    "bacc": 0.85660844727908
  },
  "devdic": {
    "prec": 0.7306843267108167,
    "rec": 0.4882005899705015,
    "f1": 0.5853227232537577,
    "bacc": 0.7301638088075034
  },
  "losses": [
    6922.2253660805,
    5005.378505286557
  ],
  "desc": "1:1 Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9527202072538861,
    "rec": 0.7362362362362362,
    "f1": 0.8306041784302655,
    "bacc": 0.8641700380748515
  },
  "devdic": {
    "prec": 0.8061224489795918,
    "rec": 0.46607669616519176,
    "f1": 0.5906542056074766,
    "bacc": 0.7243566025948189
  },
  "losses": [
    6634.815536359558,
    4379.086781309859
  ],
  "desc": "1vs1"
},
{
  "testdic": {
    "prec": 0.8879359634076616,
    "rec": 0.7772772772772772,
    "f1": 0.8289298105150789,
    "bacc": 0.878038314138909
  },
  "devdic": {
    "prec": 0.6421052631578947,
    "rec": 0.5398230088495575,
    "f1": 0.5865384615384615,
    "bacc": 0.7466078717996931
  },
  "losses": [
    6714.445379352663,
    4723.973620550707
  ],
  "desc": "1vs1"
},
{
  "testdic": {
    "prec": 0.8959810874704491,
    "rec": 0.7587587587587588,
    "f1": 0.8216802168021681,
    "bacc": 0.8698607206449283
  },
  "devdic": {
    "prec": 0.6880907372400756,
    "rec": 0.5368731563421829,
    "f1": 0.6031483015741508,
    "bacc": 0.7495880517831546
  },
  "losses": [
    6697.8648719433695,
    4706.817511115572
  ],
  "desc": "1vs1"
},
{
  "testdic": {
    "prec": 0.9262899262899262,
    "rec": 0.7547547547547547,
    "f1": 0.8317705460562602,
    "bacc": 0.8708873827857062
  },
  "devdic": {
    "prec": 0.6754850088183422,
    "rec": 0.5648967551622419,
    "f1": 0.6152610441767069,
    "bacc": 0.7614294148212397
  },
  "losses": [
    6748.4085835870355,
    4880.991608895129
  ],
  "desc": "1vs1"
},
{
  "testdic": {
    "prec": 0.967280163599182,
    "rec": 0.7102102102102102,
    "f1": 0.819047619047619,
    "bacc": 0.8525091072684366
  },
  "devdic": {
    "prec": 0.827683615819209,
    "rec": 0.43215339233038347,
    "f1": 0.5678294573643411,
    "bacc": 0.7091084530763181
  },
  "losses": [
    6775.200716448831,
    4679.330335480627
  ],
  "desc": "1vs1"
},
{
  "testdic": {
    "prec": 0.9655870445344129,
    "rec": 0.7162162162162162,
    "f1": 0.8224137931034483,
    "bacc": 0.8553498604066478
  },
  "devdic": {
    "prec": 0.8145161290322581,
    "rec": 0.4469026548672566,
    "f1": 0.5771428571428571,
    "bacc": 0.7155692163986729
  },
  "losses": [
    6793.976902139839,
    4787.059024915419
  ],
  "desc": "1vs1"
},
{
  "testdic": {
    "prec": 0.9711538461538461,
    "rec": 0.7077077077077077,
    "f1": 0.8187608569774175,
    "bacc": 0.851582355746769
  },
  "devdic": {
    "prec": 0.8205128205128205,
    "rec": 0.4247787610619469,
    "f1": 0.5597667638483965,
    "bacc": 0.7051926704555793
  },
  "losses": [
    6787.233861104469,
    4546.342129350698
  ],
  "desc": "1vs1"
},
]

single_ordering_sgd = [
{
  "testdic": {
    "prec": 0.9539776462853385,
    "rec": 0.7262262262262262,
    "f1": 0.8246660983233873,
    "bacc": 0.8593272829346383
  },
  "devdic": {
    "prec": 0.7985074626865671,
    "rec": 0.47345132743362833,
    "f1": 0.5944444444444444,
    "bacc": 0.727472750762736
  },
  "losses": [
    21895.148438185453,
    18416.61286021024
  ],
  "desc": "1vs1 ordering"
},
{
  "testdic": {
    "prec": 0.9833333333333333,
    "rec": 0.7087087087087087,
    "f1": 0.8237347294938917,
    "bacc": 0.8530563554360202
  },
  "devdic": {
    "prec": 0.8717105263157895,
    "rec": 0.39085545722713866,
    "f1": 0.539714867617108,
    "bacc": 0.6909726223764205
  },
  "losses": [
    22537.006257504225,
    19122.508245959878
  ],
  "desc": "1vs1 ordering"
},
{
  "testdic": {
    "prec": 0.9359137055837563,
    "rec": 0.7382382382382382,
    "f1": 0.8254057078903189,
    "bacc": 0.8636567070044625
  },
  "devdic": {
    "prec": 0.7313432835820896,
    "rec": 0.5058997050147492,
    "f1": 0.5980819529206626,
    "bacc": 0.7385564323565864
  },
  "losses": [
    21697.553322836757,
    18593.14772843942
  ],
  "desc": "1vs1 ordering"
},
{
  "testdic": {
    "prec": 0.9336702463676564,
    "rec": 0.7397397397397397,
    "f1": 0.825467746439542,
    "bacc": 0.8641911246021576
  },
  "devdic": {
    "prec": 0.7662921348314606,
    "rec": 0.5029498525073747,
    "f1": 0.6073018699910954,
    "bacc": 0.7395946429546241
  },
  "losses": [
    22477.574863016605,
    18791.629362255335
  ],
  "desc": "1vs1 ordering"
},
{
  "testdic": {
    "prec": 0.9420751113940166,
    "rec": 0.7407407407407407,
    "f1": 0.8293639674978986,
    "bacc": 0.865448791138353
  },
  "devdic": {
    "prec": 0.7787810383747178,
    "rec": 0.5088495575221239,
    "f1": 0.615521855486173,
    "bacc": 0.74322989642156
  },
  "losses": [
    22809.43369269371,
    19185.18961663544
  ],
  "desc": "1vs1 ordering"
},
{
  "testdic": {
    "prec": 0.9661246612466124,
    "rec": 0.7137137137137137,
    "f1": 0.8209556706966034,
    "bacc": 0.8541526924436605
  },
  "devdic": {
    "prec": 0.8452722063037249,
    "rec": 0.4351032448377581,
    "f1": 0.5744888023369036,
    "bacc": 0.711383013782827
  },
  "losses": [
    22196.403668999672,
    18694.0721546039
  ],
  "desc": "1vs1 ordering"
},
{
  "testdic": {
    "prec": 0.9716655148583275,
    "rec": 0.7037037037037037,
    "f1": 0.8162554426705371,
    "bacc": 0.8496344370330309
  },
  "devdic": {
    "prec": 0.8551136363636364,
    "rec": 0.443952802359882,
    "f1": 0.5844660194174758,
    "bacc": 0.7161504930236695
  },
  "losses": [
    22210.739150717854,
    18753.90790149197
  ],
  "desc": "1vs1 ordering"
},

]

one_vs_two_cls_sgd = [
{
  "testdic": {
    "prec": 0.9502617801047121,
    "rec": 0.7267267267267268,
    "f1": 0.8235961429381737,
    "bacc": 0.859253033455305
  },
  "devdic": {
    "prec": 0.8108108108108109,
    "rec": 0.4424778761061947,
    "f1": 0.5725190839694657,
    "bacc": 0.7132425935248816
  },
  "losses": [
    6630.2669124688255,
    4383.030740029499
  ],
  "desc": "SGD 1:2 [CLS]池化, flrate=0"
},
{
  "testdic": {
    "prec": 0.9427828348504551,
    "rec": 0.7257257257257257,
    "f1": 0.8201357466063348,
    "bacc": 0.8581035334956373
  },
  "devdic": {
    "prec": 0.7949438202247191,
    "rec": 0.4174041297935103,
    "f1": 0.5473887814313345,
    "bacc": 0.7003630198887588
  },
  "losses": [
    6766.955830838182,
    4515.771810124017
  ],
  "desc": "SGD 1:2 [CLS]池化, flrate=0"
},
{
  "testdic": {
    "prec": 0.9644533869885983,
    "rec": 0.7197197197197197,
    "f1": 0.824304958440814,
    "bacc": 0.8569934455818717
  },
  "devdic": {
    "prec": 0.7447795823665894,
    "rec": 0.47345132743362833,
    "f1": 0.5788999098286746,
    "bacc": 0.7241599794581896
  },
  "losses": [
    6701.277475580631,
    4536.742896225478
  ],
  "desc": "SGD 1:2 [CLS]池化, flrate=0"
},
{
  "testdic": {
    "prec": 0.9521625163826999,
    "rec": 0.7272272272272272,
    "f1": 0.8246311010215664,
    "bacc": 0.8596655335703469
  },
  "devdic": {
    "prec": 0.7494033412887828,
    "rec": 0.4631268436578171,
    "f1": 0.5724703737465816,
    "bacc": 0.719568905036585
  },
  "losses": [
    6675.0322380196885,
    4393.616250482766
  ],
  "desc": "SGD 1:2 [CLS]池化, flrate=0"
},
{
  "testdic": {
    "prec": 0.9523809523809523,
    "rec": 0.7307307307307307,
    "f1": 0.8269612007929764,
    "bacc": 0.8614172853220987
  },
  "devdic": {
    "prec": 0.8110831234256927,
    "rec": 0.4749262536873156,
    "f1": 0.5990697674418605,
    "bacc": 0.728895614849141
  },
  "losses": [
    6951.961472243769,
    4666.539934770437
  ],
  "desc": "SGD 1:2 [CLS]池化, flrate=0"
},
{
  "testdic": {
    "prec": 0.9614873837981408,
    "rec": 0.7247247247247247,
    "f1": 0.8264840182648401,
    "bacc": 0.8592255316430546
  },
  "devdic": {
    "prec": 0.7984886649874056,
    "rec": 0.46755162241887904,
    "f1": 0.5897674418604651,
    "bacc": 0.7246371317486215
  },
  "losses": [
    6708.058002991194,
    4451.932718374825
  ],
  "desc": "SGD 1:2 [CLS]池化, flrate=0"
},
]

ordering_only = [
{
  "testdic": {
    "prec": 0.7216269841269841,
    "rec": 0.6812137104326653,
    "f1": 0.7008382310434531,
    "bacc": 0.7217891045388258
  },
  "devdic": {
    "prec": 0.6514285714285715,
    "rec": 0.4896907216494845,
    "f1": 0.5590975968612065,
    "bacc": 0.6330008430396304
  },
  "losses": [
    19401.672472715378,
    16884.871153742075
  ],
  "desc": "2:2 Ordering Only, flrate=0"
},
]

# 反而用sector来增强ordering
ordering_with_sector = [
(0.68875, 0.6243626062322947, 0.6549777117384844, 0.6865928700293955),
(0.8187442004330343, 0.5081589556536763, 0.6271025823264629, 0.705521307458923)
]

sector_sep_ordering_cls = [
{
  "testdic": {
    "prec": 0.9600262123197904,
    "rec": 0.7332332332332332,
    "f1": 0.8314415437003405,
    "bacc": 0.8633175360325172
  },
  "devdic": {
    "prec": 0.7969543147208121,
    "rec": 0.4631268436578171,
    "f1": 0.585820895522388,
    "bacc": 0.7224247423680906
  },
  "losses": [
    24633.783616339788,
    20707.941033105366
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
}, 
{
  "testdic": {
    "prec": 0.945407835581246,
    "rec": 0.7367367367367368,
    "f1": 0.8281293952180028,
    "bacc": 0.8637712888659346
  },
  "devdic": {
    "prec": 0.810126582278481,
    "rec": 0.471976401179941,
    "f1": 0.5964585274930103,
    "bacc": 0.7274206885954537
  },
  "losses": [
    25382.128994267434,
    21237.201927867718
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9524752475247524,
    "rec": 0.7222222222222222,
    "f1": 0.8215200683176772,
    "bacc": 0.8572171143561085
  },
  "devdic": {
    "prec": 0.7709923664122137,
    "rec": 0.4469026548672566,
    "f1": 0.5658263305322129,
    "bacc": 0.7131703130402082
  },
  "losses": [
    26657.7022087276,
    22286.090343401767
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9641891891891892,
    "rec": 0.7142142142142142,
    "f1": 0.8205865439907992,
    "bacc": 0.854240692829119
  },
  "devdic": {
    "prec": 0.8857142857142857,
    "rec": 0.41150442477876104,
    "f1": 0.5619335347432024,
    "bacc": 0.7016398066320124
  },
  "losses": [
    26881.95822713524,
    22474.238678359427
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9685362517099864,
    "rec": 0.7087087087087087,
    "f1": 0.8184971098265895,
    "bacc": 0.8518665230942137
  },
  "devdic": {
    "prec": 0.7789473684210526,
    "rec": 0.4365781710914454,
    "f1": 0.5595463137996219,
    "bacc": 0.708693472111864
  },
  "losses": [
    24913.8883514395,
    21250.28221097472
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9502292075965947,
    "rec": 0.7262262262262262,
    "f1": 0.8232624113475177,
    "bacc": 0.8590027832050546
  },
  "devdic": {
    "prec": 0.8228882833787466,
    "rec": 0.44542772861356933,
    "f1": 0.5779904306220096,
    "bacc": 0.7152886872448702
  },
  "losses": [
    25434.955314576626,
    21366.486725728726
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.8985422740524781,
    "rec": 0.7712712712712713,
    "f1": 0.8300565580393213,
    "bacc": 0.8762251434777124
  },
  "devdic": {
    "prec": 0.7358490566037735,
    "rec": 0.5176991150442478,
    "f1": 0.6077922077922078,
    "bacc": 0.7444561373713356
  },
  "losses": [
    25247.986713021994,
    21552.776726243552
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.8901035673187572,
    "rec": 0.7742742742742743,
    "f1": 0.8281584582441114,
    "bacc": 0.8768072290787272
  },
  "devdic": {
    "prec": 0.7021276595744681,
    "rec": 0.5353982300884956,
    "f1": 0.6075313807531381,
    "bacc": 0.7501071570821733
  },
  "losses": [
    26061.67810959369,
    21558.116234263405
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.7946257197696737,
    "rec": 0.8288288288288288,
    "f1": 0.8113669769720725,
    "bacc": 0.8912667670374539
  },
  "devdic": {
    "prec": 0.5714285714285714,
    "rec": 0.6194690265486725,
    "f1": 0.5944798301486199,
    "bacc": 0.7737509628973658
  },
  "losses": [
    25795.727253615856,
    21795.809214139823
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},

{
  "testdic": {
    "prec": 0.9371827411167513,
    "rec": 0.7392392392392393,
    "f1": 0.8265249020705092,
    "bacc": 0.864265374081491
  },
  "devdic": {
    "prec": 0.801007556675063,
    "rec": 0.4690265486725664,
    "f1": 0.5916279069767442,
    "bacc": 0.7254888283687255
  },
  "losses": [
    25664.77320991829,
    21681.04003435839
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9052443384982122,
    "rec": 0.7602602602602603,
    "f1": 0.8264417845484222,
    "bacc": 0.8715308872961658
  },
  "devdic": {
    "prec": 0.7906403940886699,
    "rec": 0.47345132743362833,
    "f1": 0.5922509225092252,
    "bacc": 0.7270158167896952
  },
  "losses": [
    26356.28333794698,
    22007.206386728212
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.827116935483871,
    "rec": 0.8213213213213213,
    "f1": 0.8242089402310396,
    "bacc": 0.8921100927861338
  },
  "devdic": {
    "prec": 0.5852272727272727,
    "rec": 0.6076696165191741,
    "f1": 0.5962373371924746,
    "bacc": 0.7704786282276017
  },
  "losses": [
    25940.459432013333,
    21711.12191291526
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.8857634902411022,
    "rec": 0.7722722722722722,
    "f1": 0.8251336898395721,
    "bacc": 0.8753735617716147
  },
  "devdic": {
    "prec": 0.686456400742115,
    "rec": 0.5457227138643068,
    "f1": 0.6080525883319639,
    "bacc": 0.7535558965711755
  },
  "losses": [
    25695.640571480617,
    21649.139069491997
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.895040369088812,
    "rec": 0.7767767767767768,
    "f1": 0.8317256162915327,
    "bacc": 0.8785452299243538
  },
  "devdic": {
    "prec": 0.7094188376753507,
    "rec": 0.5221238938053098,
    "f1": 0.6015293118096856,
    "bacc": 0.7444980903799224
  },
  "losses": [
    26510.692346453667,
    21877.041479830164
  ],
  "desc": "2:2 Sector_SEP_Order_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.909769757311761,
    "rec": 0.7317317317317318,
    "f1": 0.8110957004160888,
    "bacc": 0.8580237890675966
  },
  "devdic": {
    "prec": 0.7246376811594203,
    "rec": 0.5162241887905604,
    "f1": 0.602928509905254,
    "bacc": 0.7429190397916705
  },
  "losses": [
    26975.588029660285,
    22833.923735845834
  ],
  "desc": "2vs2, plus ordering"
},
{
  "testdic": {
    "prec": 0.9571240105540897,
    "rec": 0.7262262262262262,
    "f1": 0.8258394991462721,
    "bacc": 0.859597699375958
  },
  "devdic": {
    "prec": 0.7890995260663507,
    "rec": 0.4911504424778761,
    "f1": 0.6054545454545455,
    "bacc": 0.7354084403387782
  },
  "losses": [
    26239.035907678306,
    21929.60196376685
  ],
  "desc": "2vs2, plus ordering"
},
{
  "testdic": {
    "prec": 0.9462086843810759,
    "rec": 0.7307307307307307,
    "f1": 0.8246258119175374,
    "bacc": 0.8608764524394594
  },
  "devdic": {
    "prec": 0.800531914893617,
    "rec": 0.443952802359882,
    "f1": 0.571157495256167,
    "bacc": 0.7134088891854242
  },
  "losses": [
    26487.87387450412,
    22106.657476269174
  ],
  "desc": "2vs2, plus ordering"
},
{
  "testdic": {
    "prec": 0.9476707083599234,
    "rec": 0.7432432432432432,
    "f1": 0.8330995792426367,
    "bacc": 0.8671867919839796
  },
  "devdic": {
    "prec": 0.8,
    "rec": 0.5014749262536873,
    "f1": 0.6165004533091568,
    "bacc": 0.7410276161997247
  },
  "losses": [
    26424.085734991357,
    22012.663512987085
  ],
  "desc": "2vs2, plus ordering"
},
{
  "testdic": {
    "prec": 0.9558047493403694,
    "rec": 0.7252252252252253,
    "f1": 0.8247011952191237,
    "bacc": 0.8589890322989295
  },
  "devdic": {
    "prec": 0.7986111111111112,
    "rec": 0.5088495575221239,
    "f1": 0.6216216216216217,
    "bacc": 0.7444864648474225
  },
  "losses": [
    25362.467258660123,
    21349.229854634963
  ],
  "desc": "2vs2, plus ordering"
},
{
  "testdic": {
    "prec": 0.9718213058419244,
    "rec": 0.7077077077077077,
    "f1": 0.8189979727772951,
    "bacc": 0.8516364390350328
  },
  "devdic": {
    "prec": 0.8668831168831169,
    "rec": 0.3938053097345133,
    "f1": 0.54158215010142,
    "bacc": 0.6922190816435875
  },
  "losses": [
    26250.531280322,
    22148.33154927753
  ],
  "desc": "2vs2, plus ordering"
},
{
  "testdic": {
    "prec": 0.9851904090267983,
    "rec": 0.6991991991991992,
    "f1": 0.8179156908665105,
    "bacc": 0.8484638505460571
  },
  "devdic": {
    "prec": 0.9035714285714286,
    "rec": 0.37315634218289084,
    "f1": 0.5281837160751566,
    "bacc": 0.6834938667734194
  },
  "losses": [
    26788.109091490507,
    22421.3425307367
  ],
  "desc": "2vs2, plus ordering"
},

]

# 有考虑SGD
double_datas = [{
  "testdic": {
    "prec": 0.9001189060642093,
    "rec": 0.7577577577577578,
    "f1": 0.8228260869565218,
    "bacc": 0.8697928864505393
  },
  "devdic": {
    "prec": 0.6545768566493955,
    "rec": 0.5589970501474927,
    "f1": 0.6030230708035005,
    "bacc": 0.7566518264217015
  },
  "losses": [
    7110.542909749434,
    5137.176332781441
  ],
  "desc": "2:2 Double_Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 1,
    "rec": 0.6266266266266266,
    "f1": 0.7704615384615385,
    "bacc": 0.8133133133133132
  },
  "devdic": {
    "prec": 0.9804878048780488,
    "rec": 0.29646017699115046,
    "f1": 0.4552661381653455,
    "bacc": 0.6477731545225343
  },
  "losses": [
    8362.54152142303,
    6388.951750492532
  ],
  "desc": "2:2 Double_Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.895662368112544,
    "rec": 0.7647647647647647,
    "f1": 0.8250539956803457,
    "bacc": 0.8727555570714034
  },
  "devdic": {
    "prec": 0.7306034482758621,
    "rec": 0.5,
    "f1": 0.5936952714535902,
    "bacc": 0.735720813342472
  },
  "losses": [
    7095.163628809038,
    7272.169616238039
  ],
  "desc": "2:2 Double_Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.9316022799240026,
    "rec": 0.7362362362362362,
    "f1": 0.8224769359798715,
    "bacc": 0.8622771229856141
  },
  "devdic": {
    "prec": 0.780952380952381,
    "rec": 0.4837758112094395,
    "f1": 0.5974499089253187,
    "bacc": 0.7313784242247792
  },
  "losses": [
    7535.903145493765,
    5441.808580169338
  ],
  "desc": "2:2 Double_Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.993421052631579,
    "rec": 0.6801801801801802,
    "f1": 0.8074866310160428,
    "bacc": 0.8396033404957148
  },
  "devdic": {
    "prec": 0.9163498098859315,
    "rec": 0.3554572271386431,
    "f1": 0.512221041445271,
    "bacc": 0.6752154767175966
  },
  "losses": [
    7174.63851182902,
    5200.972930922624
  ],
  "desc": "2:2 Double_Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.8620082207868467,
    "rec": 0.7347347347347347,
    "f1": 0.7932991083490948,
    "bacc": 0.8546577946253446
  },
  "devdic": {
    "prec": 0.5935483870967742,
    "rec": 0.5427728613569321,
    "f1": 0.5670261941448382,
    "bacc": 0.7425995903768896
  },
  "losses": [
    7409.50640684471,
    5211.504162163794
  ],
  "desc": "2:2 Double_Sentence_CLS, flrate=0"
},
{
  "testdic": {
    "prec": 0.924375,
    "rec": 0.7402402402402403,
    "f1": 0.8221234018899389,
    "bacc": 0.863576042240185
  },
  "devdic": {
    "prec": 0.7786069651741293,
    "rec": 0.4616519174041298,
    "f1": 0.5796296296296296,
    "bacc": 0.7206591778019049
  },
  "losses": [
    7002.856738464849,
    5065.29419643592
  ],
  "desc": "2vs2"
},
{
  "testdic": {
    "prec": 0.9150943396226415,
    "rec": 0.7282282282282282,
    "f1": 0.8110367892976589,
    "bacc": 0.8568128701984841
  },
  "devdic": {
    "prec": 0.6819085487077535,
    "rec": 0.5058997050147492,
    "f1": 0.5808636748518204,
    "bacc": 0.7346724935857387
  },
  "losses": [
    7395.359402891598,
    5711.642115511233
  ],
  "desc": "2vs2"
},
{
  "testdic": {
    "prec": 0.9724517906336089,
    "rec": 0.7067067067067067,
    "f1": 0.8185507246376812,
    "bacc": 0.8511900218227963
  },
  "devdic": {
    "prec": 0.8650306748466258,
    "rec": 0.415929203539823,
    "f1": 0.5617529880478088,
    "bacc": 0.7029383280664616
  },
  "losses": [
    7126.238420827314,
    5140.735351897019
  ],
  "desc": "2vs2"
},
{
  "testdic": {
    "prec": 0.8662169758291175,
    "rec": 0.7712712712712713,
    "f1": 0.815991527667461,
    "bacc": 0.8727638130288211
  },
  "devdic": {
    "prec": 0.6797020484171322,
    "rec": 0.5383480825958702,
    "f1": 0.6008230452674896,
    "bacc": 0.7495258804571766
  },
  "losses": [
    6977.6507864926825,
    5557.992111502797
  ],
  "desc": "2vs2"
},
]

# 有【符号的情况
mainichi_1vs1_1by1 = [
{
  "testdic": {
    "prec": 0.750206782464847,
    "rec": 0.557810578105781,
    "f1": 0.6398589065255732,
    "bacc": 0.7341512878673541
  },
  "losses": [
    4562.065858967602
  ],
  "desc": "1 vs 1, mainichi news epoch 1"
},
{
  "testdic": {
    "prec": 0.6725609756097561,
    "rec": 0.6783517835178352,
    "f1": 0.6754439681567668,
    "bacc": 0.75959675720053
  },
  "losses": [
    1913.2177892783657
  ],
  "desc": "1 vs 1, mainichi news epoch 2"
}
]

mainichi_1vs1_1by1_no_sepecial_char = [
{
  "testdic": {
    "prec": 0.6460078168620882,
    "rec": 0.7208722741433021,
    "f1": 0.6813898704358068,
    "bacc": 0.7670635302969824
  },
  "losses": [
    4797.132436946034
  ],
  "desc": "1 vs 1, mainichi news"
},
{
  "testdic": {
    "prec": 0.6103430619559652,
    "rec": 0.7426791277258566,
    "f1": 0.6700393479482855,
    "bacc": 0.7592629806523246
  },
  "losses": [
    2309.2673610709608
  ],
  "desc": "1 vs 1, mainichi news"
},
]

mainichi_2vs2_1by1_no_special = [
{
  "testdic": {
    "prec": 0.711352657004831,
    "rec": 0.7339563862928349,
    "f1": 0.7224777675559645,
    "bacc": 0.7965805495528975
  },
  "losses": [
    4667.068539037835
  ],
  "desc": "2 vs 2, mainichi news epoch 1"
},
{
  "testdic": {
    "prec": 0.7297633872976339,
    "rec": 0.7302180685358255,
    "f1": 0.7299906571161633,
    "bacc": 0.8011915084947169
  },
  "losses": [
    2286.724190486246
  ],
  "desc": "2 vs 2, mainichi news epoch 2"
},
]

ordering_mainichi_2vs2_1by1_no_special = [
  {'prec': 0.6554243957279371, 'rec': 0.7264797507788162, 'f1': 0.6891252955082743, 'bacc': 0.772960052119894},
  {'prec': 0.6815578465063001, 'rec': 0.7414330218068536, 'f1': 0.7102357505222321, 'bacc': 0.7888313857193325},
]

ordering_mainichi_2vs2_2poch = [
{
  "testdic": {
    "prec": 0.7082278481012658,
    "rec": 0.697196261682243,
    "f1": 0.7026687598116169,
    "bacc": 0.7807041691327268
  },
  "losses": [
    12279.389788985252,
    9447.7157625556
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6940828402366864,
    "rec": 0.7308411214953271,
    "f1": 0.7119878603945372,
    "bacc": 0.7892791763588565
  },
  "losses": [
    12237.443496346474,
    9349.254914015532
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6774547449259463,
    "rec": 0.7694704049844237,
    "f1": 0.7205367561260211,
    "bacc": 0.7981372643478819
  },
  "losses": [
    11498.994660198689,
    8808.950879752636
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6581243184296619,
    "rec": 0.75202492211838,
    "f1": 0.7019482407676649,
    "bacc": 0.783670782119573
  },
  "losses": [
    12439.263530015945,
    9787.85341565311
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6993827160493827,
    "rec": 0.7059190031152648,
    "f1": 0.7026356589147287,
    "bacc": 0.7812363793190462
  },
  "losses": [
    12200.563538610935,
    9372.28810313344
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6643394199785178,
    "rec": 0.7707165109034267,
    "f1": 0.7135852321892125,
    "bacc": 0.793311127322111
  },
  "losses": [
    12335.5635009408,
    9674.616326466203
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.673402512288367,
    "rec": 0.7682242990654206,
    "f1": 0.7176949941792783,
    "bacc": 0.7960414573383068
  },
  "losses": [
    11635.992681443691,
    8841.067985996604
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.708078335373317,
    "rec": 0.7208722741433021,
    "f1": 0.7144180302562518,
    "bacc": 0.7901857688831386
  },
  "losses": [
    11072.089094996452,
    8327.778069108725
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6881720430107527,
    "rec": 0.7576323987538941,
    "f1": 0.7212336892052194,
    "bacc": 0.7976674512178896
  },
  "losses": [
    12276.221462488174,
    9393.830524384975
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6848203939745076,
    "rec": 0.7364485981308411,
    "f1": 0.7096967877514261,
    "bacc": 0.7881064787414147
  },
  "losses": [
    12289.087785601616,
    9401.423734560609
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.7030796048808832,
    "rec": 0.7538940809968847,
    "f1": 0.7276007215874923,
    "bacc": 0.8016893085396795
  },
  "losses": [
    12156.553824841976,
    9111.019850999117
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6500769625448948,
    "rec": 0.7894080996884735,
    "f1": 0.7129994372537986,
    "bacc": 0.7942622236292147
  },
  "losses": [
    12502.560990154743,
    10036.769762575626
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.7222222222222222,
    "rec": 0.7208722741433021,
    "f1": 0.721546616775803,
    "bacc": 0.7948985818433743
  },
  "losses": [
    12131.819679796696,
    9160.419478476048
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.705775075987842,
    "rec": 0.7233644859813084,
    "f1": 0.7144615384615385,
    "bacc": 0.7904009469670901
  },
  "losses": [
    11639.243910491467,
    8800.741563528776
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
{
  "testdic": {
    "prec": 0.6923076923076923,
    "rec": 0.7289719626168224,
    "f1": 0.7101669195751137,
    "bacc": 0.787902770704582
  },
  "losses": [
    11597.446954786777,
    8780.77689704299
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},

{
  "testdic": {
    "prec": 0.7082278481012658,
    "rec": 0.697196261682243,
    "f1": 0.7026687598116169,
    "bacc": 0.7807041691327268
  },
  "losses": [
    12279.389788985252,
    9447.7157625556
  ],
  "desc": "2 vs 2 + ordering, mainichi news epoch 2, lr = 5e-6"
},
]

# 有【符号的情况
# NOTE: 下降了
mainichi_2vs2_1by1 = [
{
  "testdic": {
    "prec": 0.6968053044002411,
    "rec": 0.7109471094710947,
    "f1": 0.7038051750380518,
    "bacc": 0.7809329501119552
  },
  "losses": [
    4244.460358268581
  ],
  "desc": "2 vs 2, mainichi news"
},
{
  "testdic": {
    "prec": 0.6817087845968712,
    "rec": 0.6968019680196802,
    "f1": 0.6891727493917275,
    "bacc": 0.7700073859066985
  },
  "losses": [
    1728.0919818053953
  ],
  "desc": "2 vs 2, mainichi news"
}
]

# 分裂sector

split_sector = [
{
  "testdic": {
    "prec": 0.7620874904067536,
    "rec": 0.6190773067331671,
    "f1": 0.6831785345717234,
    "bacc": 0.763896721682249
  },
  "losses": [
    17213.20941835642
  ],
  "desc": "分裂sector"
},
{
  "testdic": {
    "prec": 0.7611395178962747,
    "rec": 0.6496259351620948,
    "f1": 0.7009754456777666,
    "bacc": 0.776668091255959
  },
  "losses": [
    8439.678816840053
  ],
  "desc": "分裂sector epoch 2"
},
{
  "testdic": {
    "prec": 0.7170762444864525,
    "rec": 0.7094763092269327,
    "f1": 0.7132560325916641,
    "bacc": 0.788630969690027
  },
  "losses": [
    3241.590220466256
  ],
  "desc": "分裂sector epoch 3"
},
{
  "testdic": {
    "prec": 0.7347498286497601,
    "rec": 0.6683291770573566,
    "f1": 0.6999673522690173,
    "bacc": 0.7771857899421059
  },
  "losses": [
    1662.0824515204877
  ],
  "desc": "分裂sector epoch 4"
},
]



def analyse(datas, dicname = 'testdic'):
  precs = []
  recs = []
  f1s = []
  baccs = []
  for data in datas:
    precs.append(data[dicname]['prec'])
    recs.append(data[dicname]['rec'])
    f1s.append(data[dicname]['f1'])
    baccs.append(data[dicname]['bacc'])
  return precs, recs, f1s, baccs

