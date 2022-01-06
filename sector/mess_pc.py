# grid search实验12.27.2021

res_pc = [[[{'f_dev': 0.6211544096103137, 'f_test': 0.6226826608505999, 'des': 'bce:number0_e0'}, {'f_dev': 0.7065514103730665, 'f_test': 0.708270998076512, 'des': 'bce:number0_e1'}, {'f_dev': 0.6637907791857643, 'f_test': 0.6785177766649975, 'des': 'bce:number0_e2'}], [{'f_dev': 0.6609508963367108, 'f_test': 0.6735576923076924, 'des': 'bce:number1_e0'}, {'f_dev': 0.6814852235412983, 'f_test': 0.6843476201752309, 'des': 'bce:number1_e1'}, {'f_dev': 0.6805283757338552, 'f_test': 0.6983764006402926, 'des': 'bce:number1_e2'}], [{'f_dev': 0.6779835390946503, 'f_test': 0.6877828054298641, 'des': 'bce:number2_e0'}, {'f_dev': 0.6420297848869277, 'f_test': 0.6651492159838138, 'des': 'bce:number2_e1'}, {'f_dev': 0.7017148014440433, 'f_test': 0.7003792667509482, 'des': 'bce:number2_e2'}], [{'f_dev': 0.6605359957961114, 'f_test': 0.6689336895797912, 'des': 'bce:number3_e0'}, {'f_dev': 0.6963426371511069, 'f_test': 0.6992345790184601, 'des': 'bce:number3_e1'}, {'f_dev': 0.669960988296489, 'f_test': 0.6750544398741835, 'des': 'bce:number3_e2'}], [{'f_dev': 0.6620944341862306, 'f_test': 0.6684822975984155, 'des': 'bce:number4_e0'}, {'f_dev': 0.6416040100250626, 'f_test': 0.6477125872318428, 'des': 'bce:number4_e1'}, {'f_dev': 0.6468540829986613, 'f_test': 0.66881667080129, 'des': 'bce:number4_e2'}], [{'f_dev': 0.6693589096415952, 'f_test': 0.6893498019109765, 'des': 'bce:number5_e0'}, {'f_dev': 0.6845222765902653, 'f_test': 0.687784977201824, 'des': 'bce:number5_e1'}, {'f_dev': 0.6765652066320218, 'f_test': 0.6981263011797363, 'des': 'bce:number5_e2'}], [{'f_dev': 0.6767186302070023, 'f_test': 0.6760895170789164, 'des': 'bce:number6_e0'}, {'f_dev': 0.6865, 'f_test': 0.6968072710323934, 'des': 'bce:number6_e1'}, {'f_dev': 0.6728596535274918, 'f_test': 0.6782936601461229, 'des': 'bce:number6_e2'}], [{'f_dev': 0.6729113924050633, 'f_test': 0.6816674561819043, 'des': 'bce:number7_e0'}, {'f_dev': 0.673147661586968, 'f_test': 0.6883969837022622, 'des': 'bce:number7_e1'}, {'f_dev': 0.6793279766252739, 'f_test': 0.6936468460321049, 'des': 'bce:number7_e2'}], [{'f_dev': 0.6540483701366981, 'f_test': 0.6658607350096711, 'des': 'bce:number8_e0'}, {'f_dev': 0.6799897514732257, 'f_test': 0.6731730540347536, 'des': 'bce:number8_e1'}, {'f_dev': 0.6632707774798927, 'f_test': 0.6605550049554013, 'des': 'bce:number8_e2'}], [{'f_dev': 0.6491180461329714, 'f_test': 0.6567615210274489, 'des': 'bce:number9_e0'}, {'f_dev': 0.6852269877100577, 'f_test': 0.6918618131231161, 'des': 'bce:number9_e1'}, {'f_dev': 0.671967171069505, 'f_test': 0.6665056749577397, 'des': 'bce:number9_e2'}]], [[[{'f_dev': 0.674710662398424, 'f_test': 0.6786289394985047, 'des': 'aux:aux0.0_number0_e0'}, {'f_dev': 0.6856264411990777, 'f_test': 0.6902316694530689, 'des': 'aux:aux0.0_number0_e1'}, {'f_dev': 0.6847339632023869, 'f_test': 0.6983189459336665, 'des': 'aux:aux0.0_number0_e2'}], [{'f_dev': 0.6743785380260892, 'f_test': 0.6803410924176078, 'des': 'aux:aux0.0_number1_e0'}, {'f_dev': 0.6474147414741475, 'f_test': 0.6595474192728198, 'des': 'aux:aux0.0_number1_e1'}, {'f_dev': 0.6917191719171917, 'f_test': 0.709460876861758, 'des': 'aux:aux0.0_number1_e2'}], [{'f_dev': 0.6300056401579244, 'f_test': 0.6340447683498178, 'des': 'aux:aux0.0_number2_e0'}, {'f_dev': 0.6923832923832923, 'f_test': 0.6944508404328805, 'des': 'aux:aux0.0_number2_e1'}, {'f_dev': 0.6727084412221647, 'f_test': 0.672824501701507, 'des': 'aux:aux0.0_number2_e2'}], [{'f_dev': 0.6384901648059543, 'f_test': 0.6194160219615672, 'des': 'aux:aux0.0_number3_e0'}, {'f_dev': 0.6915622784211771, 'f_test': 0.6925438596491228, 'des': 'aux:aux0.0_number3_e1'}, {'f_dev': 0.668677727501257, 'f_test': 0.6835855646100117, 'des': 'aux:aux0.0_number3_e2'}], [{'f_dev': 0.704712289841345, 'f_test': 0.7003546099290779, 'des': 'aux:aux0.0_number4_e0'}, {'f_dev': 0.6937325551890382, 'f_test': 0.6988917708087714, 'des': 'aux:aux0.0_number4_e1'}, {'f_dev': 0.6857424765663541, 'f_test': 0.6919018123422803, 'des': 'aux:aux0.0_number4_e2'}], [{'f_dev': 0.623574144486692, 'f_test': 0.6306355241716458, 'des': 'aux:aux0.0_number5_e0'}, {'f_dev': 0.6714097496706193, 'f_test': 0.6752031519330214, 'des': 'aux:aux0.0_number5_e1'}, {'f_dev': 0.6874257190396957, 'f_test': 0.7018633540372671, 'des': 'aux:aux0.0_number5_e2'}], [{'f_dev': 0.6009389671361502, 'f_test': 0.6181721289634959, 'des': 'aux:aux0.0_number6_e0'}, {'f_dev': 0.6598602901665771, 'f_test': 0.6676737160120846, 'des': 'aux:aux0.0_number6_e1'}, {'f_dev': 0.6478332419089413, 'f_test': 0.6642893530310748, 'des': 'aux:aux0.0_number6_e2'}], [{'f_dev': 0.6331428571428571, 'f_test': 0.6365087812666311, 'des': 'aux:aux0.0_number7_e0'}, {'f_dev': 0.6723223753976671, 'f_test': 0.6790697674418604, 'des': 'aux:aux0.0_number7_e1'}, {'f_dev': 0.6874848704914065, 'f_test': 0.6983842010771993, 'des': 'aux:aux0.0_number7_e2'}], [{'f_dev': 0.6759236300520703, 'f_test': 0.6840646651270208, 'des': 'aux:aux0.0_number8_e0'}, {'f_dev': 0.6573275862068966, 'f_test': 0.6683391871550426, 'des': 'aux:aux0.0_number8_e1'}, {'f_dev': 0.6908064895367976, 'f_test': 0.6987686895338611, 'des': 'aux:aux0.0_number8_e2'}], [{'f_dev': 0.6525759577278731, 'f_test': 0.6594594594594595, 'des': 'aux:aux0.0_number9_e0'}, {'f_dev': 0.6946194861851673, 'f_test': 0.6957116788321168, 'des': 'aux:aux0.0_number9_e1'}, {'f_dev': 0.6676946800308403, 'f_test': 0.6858913250714966, 'des': 'aux:aux0.0_number9_e2'}]], [[{'f_dev': 0.6499472016895459, 'f_test': 0.6651960784313725, 'des': 'aux:aux0.1_number0_e0'}, {'f_dev': 0.6913082437275985, 'f_test': 0.7076531693873224, 'des': 'aux:aux0.1_number0_e1'}, {'f_dev': 0.6675257731958762, 'f_test': 0.6889312977099238, 'des': 'aux:aux0.1_number0_e2'}], [{'f_dev': 0.6748991935483871, 'f_test': 0.6931711880261928, 'des': 'aux:aux0.1_number1_e0'}, {'f_dev': 0.6871046228710462, 'f_test': 0.7006137758581497, 'des': 'aux:aux0.1_number1_e1'}, {'f_dev': 0.6847669389716482, 'f_test': 0.6978513876454789, 'des': 'aux:aux0.1_number1_e2'}], [{'f_dev': 0.6351931330472104, 'f_test': 0.6391096979332272, 'des': 'aux:aux0.1_number2_e0'}, {'f_dev': 0.6858928142273492, 'f_test': 0.7017780778753094, 'des': 'aux:aux0.1_number2_e1'}, {'f_dev': 0.6697650400206558, 'f_test': 0.6828687157493448, 'des': 'aux:aux0.1_number2_e2'}], [{'f_dev': 0.668217054263566, 'f_test': 0.6732245681381959, 'des': 'aux:aux0.1_number3_e0'}, {'f_dev': 0.6932606541129831, 'f_test': 0.704051270313573, 'des': 'aux:aux0.1_number3_e1'}, {'f_dev': 0.6931688371030387, 'f_test': 0.7027942421676546, 'des': 'aux:aux0.1_number3_e2'}], [{'f_dev': 0.6472046268245663, 'f_test': 0.6575342465753425, 'des': 'aux:aux0.1_number4_e0'}, {'f_dev': 0.6886301709189993, 'f_test': 0.6939814814814814, 'des': 'aux:aux0.1_number4_e1'}, {'f_dev': 0.6742366893767348, 'f_test': 0.6847648022466651, 'des': 'aux:aux0.1_number4_e2'}], [{'f_dev': 0.6664937759336099, 'f_test': 0.6766844401357247, 'des': 'aux:aux0.1_number5_e0'}, {'f_dev': 0.6529717885510818, 'f_test': 0.6668367346938776, 'des': 'aux:aux0.1_number5_e1'}, {'f_dev': 0.6829025844930419, 'f_test': 0.6858202038924931, 'des': 'aux:aux0.1_number5_e2'}], [{'f_dev': 0.686734693877551, 'f_test': 0.6859171597633136, 'des': 'aux:aux0.1_number6_e0'}, {'f_dev': 0.6725941422594143, 'f_test': 0.6837000236574403, 'des': 'aux:aux0.1_number6_e1'}, {'f_dev': 0.6828901154039136, 'f_test': 0.6936685288640596, 'des': 'aux:aux0.1_number6_e2'}], [{'f_dev': 0.6471897607122983, 'f_test': 0.6567010309278349, 'des': 'aux:aux0.1_number7_e0'}, {'f_dev': 0.6902160101651843, 'f_test': 0.6953831508805332, 'des': 'aux:aux0.1_number7_e1'}, {'f_dev': 0.6841085271317829, 'f_test': 0.7040216900135563, 'des': 'aux:aux0.1_number7_e2'}], [{'f_dev': 0.6710735060814383, 'f_test': 0.6714145383104125, 'des': 'aux:aux0.1_number8_e0'}, {'f_dev': 0.6765720350225524, 'f_test': 0.6888667992047713, 'des': 'aux:aux0.1_number8_e1'}, {'f_dev': 0.6483164083377871, 'f_test': 0.6726146220570013, 'des': 'aux:aux0.1_number8_e2'}], [{'f_dev': 0.6529335071707952, 'f_test': 0.6666666666666667, 'des': 'aux:aux0.1_number9_e0'}, {'f_dev': 0.6409747803910457, 'f_test': 0.6453149814716782, 'des': 'aux:aux0.1_number9_e1'}, {'f_dev': 0.6763580719204284, 'f_test': 0.6868782567503553, 'des': 'aux:aux0.1_number9_e2'}]], [[{'f_dev': 0.6803774497943382, 'f_test': 0.6940990278091793, 'des': 'aux:aux0.2_number0_e0'}, {'f_dev': 0.6814666999251684, 'f_test': 0.6919518963922294, 'des': 'aux:aux0.2_number0_e1'}, {'f_dev': 0.6779661016949153, 'f_test': 0.6973387460532251, 'des': 'aux:aux0.2_number0_e2'}], [{'f_dev': 0.6735007688364941, 'f_test': 0.6804123711340205, 'des': 'aux:aux0.2_number1_e0'}, {'f_dev': 0.6956727358713644, 'f_test': 0.7102434744461505, 'des': 'aux:aux0.2_number1_e1'}, {'f_dev': 0.662148070907195, 'f_test': 0.6858513189448441, 'des': 'aux:aux0.2_number1_e2'}], [{'f_dev': 0.6549160011016248, 'f_test': 0.6661569826707441, 'des': 'aux:aux0.2_number2_e0'}, {'f_dev': 0.6455661664392907, 'f_test': 0.6631446540880503, 'des': 'aux:aux0.2_number2_e1'}, {'f_dev': 0.6765799256505576, 'f_test': 0.6894508999772158, 'des': 'aux:aux0.2_number2_e2'}], [{'f_dev': 0.6469613259668509, 'f_test': 0.6670055922724963, 'des': 'aux:aux0.2_number3_e0'}, {'f_dev': 0.6955679341244855, 'f_test': 0.6932377966860725, 'des': 'aux:aux0.2_number3_e1'}, {'f_dev': 0.6608511763150938, 'f_test': 0.6833414278776105, 'des': 'aux:aux0.2_number3_e2'}], [{'f_dev': 0.6659691341878107, 'f_test': 0.6824969400244798, 'des': 'aux:aux0.2_number4_e0'}, {'f_dev': 0.6961900049480455, 'f_test': 0.6977283263792304, 'des': 'aux:aux0.2_number4_e1'}, {'f_dev': 0.6725297465080187, 'f_test': 0.6908917045182884, 'des': 'aux:aux0.2_number4_e2'}], [{'f_dev': 0.6888888888888889, 'f_test': 0.7074050498620836, 'des': 'aux:aux0.2_number5_e0'}, {'f_dev': 0.6987607244995234, 'f_test': 0.7049910873440285, 'des': 'aux:aux0.2_number5_e1'}, {'f_dev': 0.6838740232062515, 'f_test': 0.6909737661182748, 'des': 'aux:aux0.2_number5_e2'}], [{'f_dev': 0.6763164462191166, 'f_test': 0.6859813084112149, 'des': 'aux:aux0.2_number6_e0'}, {'f_dev': 0.674473067915691, 'f_test': 0.6912015598342676, 'des': 'aux:aux0.2_number6_e1'}, {'f_dev': 0.6721629485935984, 'f_test': 0.7071748878923767, 'des': 'aux:aux0.2_number6_e2'}], [{'f_dev': 0.6793139293139292, 'f_test': 0.6823472356935014, 'des': 'aux:aux0.2_number7_e0'}, {'f_dev': 0.6768138001014714, 'f_test': 0.7027790861987753, 'des': 'aux:aux0.2_number7_e1'}, {'f_dev': 0.6617762640817396, 'f_test': 0.6808305166586189, 'des': 'aux:aux0.2_number7_e2'}], [{'f_dev': 0.6587664960948021, 'f_test': 0.6670007516913053, 'des': 'aux:aux0.2_number8_e0'}, {'f_dev': 0.6897386253630204, 'f_test': 0.7058558558558559, 'des': 'aux:aux0.2_number8_e1'}, {'f_dev': 0.6530082987551867, 'f_test': 0.6775982638051604, 'des': 'aux:aux0.2_number8_e2'}], [{'f_dev': 0.6895033031563493, 'f_test': 0.6948271967472329, 'des': 'aux:aux0.2_number9_e0'}, {'f_dev': 0.6863033873343152, 'f_test': 0.6965105601469238, 'des': 'aux:aux0.2_number9_e1'}, {'f_dev': 0.6661538461538462, 'f_test': 0.6736842105263158, 'des': 'aux:aux0.2_number9_e2'}]], [[{'f_dev': 0.6807228915662651, 'f_test': 0.6938964957066605, 'des': 'aux:aux0.3_number0_e0'}, {'f_dev': 0.6852728631946176, 'f_test': 0.6949270326615705, 'des': 'aux:aux0.3_number0_e1'}, {'f_dev': 0.6815317498788172, 'f_test': 0.6923944933423607, 'des': 'aux:aux0.3_number0_e2'}], [{'f_dev': 0.6795105441291331, 'f_test': 0.6919932676124069, 'des': 'aux:aux0.3_number1_e0'}, {'f_dev': 0.6838178543864163, 'f_test': 0.6979707409155262, 'des': 'aux:aux0.3_number1_e1'}, {'f_dev': 0.6817645671148856, 'f_test': 0.6954035627886519, 'des': 'aux:aux0.3_number1_e2'}], [{'f_dev': 0.6857284932869219, 'f_test': 0.6925207756232687, 'des': 'aux:aux0.3_number2_e0'}, {'f_dev': 0.6976083353066539, 'f_test': 0.6944747761519983, 'des': 'aux:aux0.3_number2_e1'}, {'f_dev': 0.6758379189594798, 'f_test': 0.6970324361628709, 'des': 'aux:aux0.3_number2_e2'}], [{'f_dev': 0.6751691827173347, 'f_test': 0.6795646916565901, 'des': 'aux:aux0.3_number3_e0'}, {'f_dev': 0.6889906046735726, 'f_test': 0.6971658112028565, 'des': 'aux:aux0.3_number3_e1'}, {'f_dev': 0.6635329045027214, 'f_test': 0.691516121655614, 'des': 'aux:aux0.3_number3_e2'}], [{'f_dev': 0.6832884097035039, 'f_test': 0.6899488926746166, 'des': 'aux:aux0.3_number4_e0'}, {'f_dev': 0.6415405777166436, 'f_test': 0.6560283687943262, 'des': 'aux:aux0.3_number4_e1'}, {'f_dev': 0.6472120896300156, 'f_test': 0.6782899450680678, 'des': 'aux:aux0.3_number4_e2'}], [{'f_dev': 0.6857868020304568, 'f_test': 0.6813291895768587, 'des': 'aux:aux0.3_number5_e0'}, {'f_dev': 0.6709870388833501, 'f_test': 0.6928225248096007, 'des': 'aux:aux0.3_number5_e1'}, {'f_dev': 0.6826523122094446, 'f_test': 0.69068660774983, 'des': 'aux:aux0.3_number5_e2'}], [{'f_dev': 0.6664866324601675, 'f_test': 0.6676684197345355, 'des': 'aux:aux0.3_number6_e0'}, {'f_dev': 0.6742443812968226, 'f_test': 0.6873035066505442, 'des': 'aux:aux0.3_number6_e1'}, {'f_dev': 0.659041394335512, 'f_test': 0.665315429440081, 'des': 'aux:aux0.3_number6_e2'}], [{'f_dev': 0.6639522258414766, 'f_test': 0.676412289395441, 'des': 'aux:aux0.3_number7_e0'}, {'f_dev': 0.6981677917068467, 'f_test': 0.7086859688195991, 'des': 'aux:aux0.3_number7_e1'}, {'f_dev': 0.6879618593563767, 'f_test': 0.7006227758007118, 'des': 'aux:aux0.3_number7_e2'}], [{'f_dev': 0.6385239027117696, 'f_test': 0.6426332288401254, 'des': 'aux:aux0.3_number8_e0'}, {'f_dev': 0.6918357715903728, 'f_test': 0.7067603160667252, 'des': 'aux:aux0.3_number8_e1'}, {'f_dev': 0.6859133897461424, 'f_test': 0.6998841251448437, 'des': 'aux:aux0.3_number8_e2'}], [{'f_dev': 0.689604325387073, 'f_test': 0.6956120092378754, 'des': 'aux:aux0.3_number9_e0'}, {'f_dev': 0.6799106034268686, 'f_test': 0.682915397234591, 'des': 'aux:aux0.3_number9_e1'}, {'f_dev': 0.6788410031653275, 'f_test': 0.6912630149388864, 'des': 'aux:aux0.3_number9_e2'}]]], [[[{'f_dev': 0.6236621347989585, 'f_test': 0.6297093649085037, 'des': 'fl:fl0.5_number0_e0'}, {'f_dev': 0.6954481464101361, 'f_test': 0.7060373216245883, 'des': 'fl:fl0.5_number0_e1'}, {'f_dev': 0.6901997213190896, 'f_test': 0.7125488493269649, 'des': 'fl:fl0.5_number0_e2'}], [{'f_dev': 0.600358422939068, 'f_test': 0.6161952301719357, 'des': 'fl:fl0.5_number1_e0'}, {'f_dev': 0.6839530332681018, 'f_test': 0.6906014177909903, 'des': 'fl:fl0.5_number1_e1'}, {'f_dev': 0.6942660004758506, 'f_test': 0.695671675263512, 'des': 'fl:fl0.5_number1_e2'}], [{'f_dev': 0.6117445838084378, 'f_test': 0.6260961998405529, 'des': 'fl:fl0.5_number2_e0'}, {'f_dev': 0.6280851063829787, 'f_test': 0.6379627198739827, 'des': 'fl:fl0.5_number2_e1'}, {'f_dev': 0.6869565217391305, 'f_test': 0.7003838338225334, 'des': 'fl:fl0.5_number2_e2'}], [{'f_dev': 0.6753512132822478, 'f_test': 0.6747445949156569, 'des': 'fl:fl0.5_number3_e0'}, {'f_dev': 0.6916588566073103, 'f_test': 0.6991618879576533, 'des': 'fl:fl0.5_number3_e1'}, {'f_dev': 0.6937304828248859, 'f_test': 0.6902296262944619, 'des': 'fl:fl0.5_number3_e2'}], [{'f_dev': 0.6661526599845797, 'f_test': 0.6736094674556212, 'des': 'fl:fl0.5_number4_e0'}, {'f_dev': 0.6726943942133816, 'f_test': 0.6932251908396947, 'des': 'fl:fl0.5_number4_e1'}, {'f_dev': 0.6299212598425197, 'f_test': 0.6685494223363287, 'des': 'fl:fl0.5_number4_e2'}], [{'f_dev': 0.6463314097279472, 'f_test': 0.6414609053497943, 'des': 'fl:fl0.5_number5_e0'}, {'f_dev': 0.6593288590604026, 'f_test': 0.6666666666666667, 'des': 'fl:fl0.5_number5_e1'}, {'f_dev': 0.6785088816612459, 'f_test': 0.684407096171802, 'des': 'fl:fl0.5_number5_e2'}], [{'f_dev': 0.6852663131196679, 'f_test': 0.7024243724522634, 'des': 'fl:fl0.5_number6_e0'}, {'f_dev': 0.6784797123780174, 'f_test': 0.6922155688622754, 'des': 'fl:fl0.5_number6_e1'}, {'f_dev': 0.6501227161167167, 'f_test': 0.665160642570281, 'des': 'fl:fl0.5_number6_e2'}], [{'f_dev': 0.6656488549618321, 'f_test': 0.6827880512091037, 'des': 'fl:fl0.5_number7_e0'}, {'f_dev': 0.6854130052724078, 'f_test': 0.6937720329024676, 'des': 'fl:fl0.5_number7_e1'}, {'f_dev': 0.7009602194787381, 'f_test': 0.6989708404802745, 'des': 'fl:fl0.5_number7_e2'}], [{'f_dev': 0.6286366229321163, 'f_test': 0.636842105263158, 'des': 'fl:fl0.5_number8_e0'}, {'f_dev': 0.6835568802781917, 'f_test': 0.7001620745542949, 'des': 'fl:fl0.5_number8_e1'}, {'f_dev': 0.6703939008894537, 'f_test': 0.6887013595874355, 'des': 'fl:fl0.5_number8_e2'}], [{'f_dev': 0.6648779178964314, 'f_test': 0.6744582420258096, 'des': 'fl:fl0.5_number9_e0'}, {'f_dev': 0.6845472440944882, 'f_test': 0.6954921803127875, 'des': 'fl:fl0.5_number9_e1'}, {'f_dev': 0.6622313203684748, 'f_test': 0.6824940047961631, 'des': 'fl:fl0.5_number9_e2'}]], [[{'f_dev': 0.6964161849710983, 'f_test': 0.6946366782006921, 'des': 'fl:fl1.0_number0_e0'}, {'f_dev': 0.7023080546396608, 'f_test': 0.6926829268292682, 'des': 'fl:fl1.0_number0_e1'}, {'f_dev': 0.6153402537485583, 'f_test': 0.6383664696399786, 'des': 'fl:fl1.0_number0_e2'}], [{'f_dev': 0.6291106662853875, 'f_test': 0.6525647805393971, 'des': 'fl:fl1.0_number1_e0'}, {'f_dev': 0.6164978292329957, 'f_test': 0.6471683063015156, 'des': 'fl:fl1.0_number1_e1'}, {'f_dev': 0.6888260254596889, 'f_test': 0.7053886925795053, 'des': 'fl:fl1.0_number1_e2'}], [{'f_dev': 0.6584251545283525, 'f_test': 0.6552849610846095, 'des': 'fl:fl1.0_number2_e0'}, {'f_dev': 0.6861715428857215, 'f_test': 0.6935745766643471, 'des': 'fl:fl1.0_number2_e1'}, {'f_dev': 0.6638874413757165, 'f_test': 0.6812289966394623, 'des': 'fl:fl1.0_number2_e2'}], [{'f_dev': 0.6703210649960847, 'f_test': 0.662414131501472, 'des': 'fl:fl1.0_number3_e0'}, {'f_dev': 0.6916940403566401, 'f_test': 0.7043611658996275, 'des': 'fl:fl1.0_number3_e1'}, {'f_dev': 0.6765381669645137, 'f_test': 0.6960435915659797, 'des': 'fl:fl1.0_number3_e2'}], [{'f_dev': 0.6075798269173381, 'f_test': 0.611064776202391, 'des': 'fl:fl1.0_number4_e0'}, {'f_dev': 0.6900866217516843, 'f_test': 0.7068654019873533, 'des': 'fl:fl1.0_number4_e1'}, {'f_dev': 0.670618556701031, 'f_test': 0.6821779803310146, 'des': 'fl:fl1.0_number4_e2'}], [{'f_dev': 0.6788211788211788, 'f_test': 0.6980356327089995, 'des': 'fl:fl1.0_number5_e0'}, {'f_dev': 0.6729215532968466, 'f_test': 0.682748538011696, 'des': 'fl:fl1.0_number5_e1'}, {'f_dev': 0.6474703982777179, 'f_test': 0.6634542226730916, 'des': 'fl:fl1.0_number5_e2'}], [{'f_dev': 0.6801848049281314, 'f_test': 0.6802656546489564, 'des': 'fl:fl1.0_number6_e0'}, {'f_dev': 0.6663121510236639, 'f_test': 0.6741460982298679, 'des': 'fl:fl1.0_number6_e1'}, {'f_dev': 0.6916666666666667, 'f_test': 0.7033512618949112, 'des': 'fl:fl1.0_number6_e2'}], [{'f_dev': 0.6504592112371691, 'f_test': 0.6475037821482602, 'des': 'fl:fl1.0_number7_e0'}, {'f_dev': 0.6841454365570121, 'f_test': 0.6911357340720221, 'des': 'fl:fl1.0_number7_e1'}, {'f_dev': 0.6816008054366978, 'f_test': 0.6866729678638941, 'des': 'fl:fl1.0_number7_e2'}], [{'f_dev': 0.6474114441416895, 'f_test': 0.6648351648351648, 'des': 'fl:fl1.0_number8_e0'}, {'f_dev': 0.6508287292817679, 'f_test': 0.6594315245478035, 'des': 'fl:fl1.0_number8_e1'}, {'f_dev': 0.6942981451797574, 'f_test': 0.7046431642304385, 'des': 'fl:fl1.0_number8_e2'}], [{'f_dev': 0.6442916093535076, 'f_test': 0.6547379032258065, 'des': 'fl:fl1.0_number9_e0'}, {'f_dev': 0.6840260390585879, 'f_test': 0.6927192370318679, 'des': 'fl:fl1.0_number9_e1'}, {'f_dev': 0.6819672131147542, 'f_test': 0.6935258500232883, 'des': 'fl:fl1.0_number9_e2'}]], [[{'f_dev': 0.6845794392523366, 'f_test': 0.6963156747329408, 'des': 'fl:fl2.0_number0_e0'}, {'f_dev': 0.6835879743716117, 'f_test': 0.6933830382106244, 'des': 'fl:fl2.0_number0_e1'}, {'f_dev': 0.6745638595735517, 'f_test': 0.6971916971916972, 'des': 'fl:fl2.0_number0_e2'}], [{'f_dev': 0.6760847628657921, 'f_test': 0.6876320112649613, 'des': 'fl:fl2.0_number1_e0'}, {'f_dev': 0.6832012273075939, 'f_test': 0.6993572958819329, 'des': 'fl:fl2.0_number1_e1'}, {'f_dev': 0.6823760432007855, 'f_test': 0.6969353007945517, 'des': 'fl:fl2.0_number1_e2'}], [{'f_dev': 0.6520654283337954, 'f_test': 0.6499741868869385, 'des': 'fl:fl2.0_number2_e0'}, {'f_dev': 0.6768767260858649, 'f_test': 0.6916666666666667, 'des': 'fl:fl2.0_number2_e1'}, {'f_dev': 0.6973929236499068, 'f_test': 0.7041036717062634, 'des': 'fl:fl2.0_number2_e2'}], [{'f_dev': 0.6784266984505363, 'f_test': 0.6946803755029056, 'des': 'fl:fl2.0_number3_e0'}, {'f_dev': 0.6576234400434074, 'f_test': 0.6804380288700848, 'des': 'fl:fl2.0_number3_e1'}, {'f_dev': 0.6865820263878518, 'f_test': 0.6980609418282547, 'des': 'fl:fl2.0_number3_e2'}], [{'f_dev': 0.695139911634757, 'f_test': 0.6938305709023941, 'des': 'fl:fl2.0_number4_e0'}, {'f_dev': 0.687591956841589, 'f_test': 0.6884422110552764, 'des': 'fl:fl2.0_number4_e1'}, {'f_dev': 0.6845540246555476, 'f_test': 0.682659355723098, 'des': 'fl:fl2.0_number4_e2'}], [{'f_dev': 0.6812484639960679, 'f_test': 0.6913013541427587, 'des': 'fl:fl2.0_number5_e0'}, {'f_dev': 0.6710114702815432, 'f_test': 0.6911095135394201, 'des': 'fl:fl2.0_number5_e1'}, {'f_dev': 0.674846625766871, 'f_test': 0.689753320683112, 'des': 'fl:fl2.0_number5_e2'}], [{'f_dev': 0.6535982814178304, 'f_test': 0.6602774274905422, 'des': 'fl:fl2.0_number6_e0'}, {'f_dev': 0.673537748003092, 'f_test': 0.6812921890067504, 'des': 'fl:fl2.0_number6_e1'}, {'f_dev': 0.6750693219057222, 'f_test': 0.6831916902738434, 'des': 'fl:fl2.0_number6_e2'}], [{'f_dev': 0.6780684104627768, 'f_test': 0.6846045858680393, 'des': 'fl:fl2.0_number7_e0'}, {'f_dev': 0.6888672824501703, 'f_test': 0.6952770208900999, 'des': 'fl:fl2.0_number7_e1'}, {'f_dev': 0.6868395773294909, 'f_test': 0.6901889369451399, 'des': 'fl:fl2.0_number7_e2'}], [{'f_dev': 0.6731578947368422, 'f_test': 0.6794966691339748, 'des': 'fl:fl2.0_number8_e0'}, {'f_dev': 0.6761584260937096, 'f_test': 0.6866315285098227, 'des': 'fl:fl2.0_number8_e1'}, {'f_dev': 0.6790464240903389, 'f_test': 0.6948356807511737, 'des': 'fl:fl2.0_number8_e2'}], [{'f_dev': 0.6452513966480448, 'f_test': 0.6558725263428425, 'des': 'fl:fl2.0_number9_e0'}, {'f_dev': 0.659504132231405, 'f_test': 0.673582995951417, 'des': 'fl:fl2.0_number9_e1'}, {'f_dev': 0.6655948553054662, 'f_test': 0.677173110071411, 'des': 'fl:fl2.0_number9_e2'}]], [[{'f_dev': 0.6621124540199684, 'f_test': 0.6769601930036189, 'des': 'fl:fl5.0_number0_e0'}, {'f_dev': 0.6639978649586336, 'f_test': 0.6726591760299625, 'des': 'fl:fl5.0_number0_e1'}, {'f_dev': 0.7010131493856435, 'f_test': 0.7037186742118028, 'des': 'fl:fl5.0_number0_e2'}], [{'f_dev': 0.6502835538752363, 'f_test': 0.6658354114713217, 'des': 'fl:fl5.0_number1_e0'}, {'f_dev': 0.6574509261819187, 'f_test': 0.6746496815286623, 'des': 'fl:fl5.0_number1_e1'}, {'f_dev': 0.6610079575596817, 'f_test': 0.6756889763779528, 'des': 'fl:fl5.0_number1_e2'}], [{'f_dev': 0.652694610778443, 'f_test': 0.6461928934010152, 'des': 'fl:fl5.0_number2_e0'}, {'f_dev': 0.651912568306011, 'f_test': 0.6707348080345792, 'des': 'fl:fl5.0_number2_e1'}, {'f_dev': 0.6919964028776978, 'f_test': 0.701732414944688, 'des': 'fl:fl5.0_number2_e2'}], [{'f_dev': 0.6815642458100559, 'f_test': 0.6941391941391941, 'des': 'fl:fl5.0_number3_e0'}, {'f_dev': 0.6954503249767873, 'f_test': 0.7028670721112076, 'des': 'fl:fl5.0_number3_e1'}, {'f_dev': 0.6696546269443712, 'f_test': 0.6684796044499381, 'des': 'fl:fl5.0_number3_e2'}], [{'f_dev': 0.671460451276965, 'f_test': 0.6795313576843556, 'des': 'fl:fl5.0_number4_e0'}, {'f_dev': 0.6746361746361745, 'f_test': 0.6817962337035248, 'des': 'fl:fl5.0_number4_e1'}, {'f_dev': 0.6699153629135676, 'f_test': 0.6769669598288567, 'des': 'fl:fl5.0_number4_e2'}], [{'f_dev': 0.5682610072853975, 'f_test': 0.5805707696742578, 'des': 'fl:fl5.0_number5_e0'}, {'f_dev': 0.7003109303994259, 'f_test': 0.7056451612903226, 'des': 'fl:fl5.0_number5_e1'}, {'f_dev': 0.6774661508704063, 'f_test': 0.6881525192918748, 'des': 'fl:fl5.0_number5_e2'}], [{'f_dev': 0.6017000607164542, 'f_test': 0.6185852981969487, 'des': 'fl:fl5.0_number6_e0'}, {'f_dev': 0.6631695196024296, 'f_test': 0.6703967446592064, 'des': 'fl:fl5.0_number6_e1'}, {'f_dev': 0.6813458683819891, 'f_test': 0.6842226842226842, 'des': 'fl:fl5.0_number6_e2'}], [{'f_dev': 0.6877526753864447, 'f_test': 0.6921538119582129, 'des': 'fl:fl5.0_number7_e0'}, {'f_dev': 0.6946975005839757, 'f_test': 0.6858776946766387, 'des': 'fl:fl5.0_number7_e1'}, {'f_dev': 0.6704486848891181, 'f_test': 0.6840718562874251, 'des': 'fl:fl5.0_number7_e2'}], [{'f_dev': 0.6708463949843261, 'f_test': 0.6798839458413927, 'des': 'fl:fl5.0_number8_e0'}, {'f_dev': 0.6911416390336333, 'f_test': 0.6991189427312775, 'des': 'fl:fl5.0_number8_e1'}, {'f_dev': 0.6745079212674028, 'f_test': 0.6943328871039715, 'des': 'fl:fl5.0_number8_e2'}], [{'f_dev': 0.6149323927101704, 'f_test': 0.6227285055600759, 'des': 'fl:fl5.0_number9_e0'}, {'f_dev': 0.6705446853516658, 'f_test': 0.6800587947084763, 'des': 'fl:fl5.0_number9_e1'}, {'f_dev': 0.6841046277665996, 'f_test': 0.6848927645533821, 'des': 'fl:fl5.0_number9_e2'}]]]]
