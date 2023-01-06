# First code extract from task pdf

import spacy

nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# cat-monkey similarity 0.5929930274321619
# banana-monkey similarity 0.40415016164997786
# banana-cat similarity 0.22358825939615987
# Interesting that it shows similarity between cat-monkey due to both being animals, but also shows a higher
# similarity between banana-monkey than for banana-cat. This suggests that spacy can infer a relationship between
# monkey and banana (at least more so than banana and cat).

# Example
my_word1 = nlp("bicycle")
my_word2 = nlp("car")
my_word3 = nlp("petrol")
print(f"{my_word1} - {my_word2} similarity: {my_word1.similarity(my_word2)}")
print(f"{my_word1} - {my_word3} similarity: {my_word1.similarity(my_word3)}")
print(f"{my_word2} - {my_word3} similarity: {my_word2.similarity(my_word3)}")

# Second code extract from task pdf
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Third code extract from task pdf
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Comments for task:
# Run the example file with the simpler language model ‘en_core_web_sm’
# and write a note on what you notice is different from the model
# 'en_core_web_md'

# Output with en_core_web_md
# -------------Complaints similarity---------------
# 1.0
# 0.835077095517691
# 0.9246800668484182
# 0.8959758607031337
# 0.8395325736974772
# 0.8622109066850939
# 0.835077095517691
# 1.0
# 0.8906698507861426
# 0.8145960252423514
# 0.9506983440908762
# 0.794650864741391
# 0.9246800668484182
# 0.8906698507861426
# 1.0
# 0.8905291077014569
# 0.88184629454754
# 0.8692563822438262
# 0.8959758607031337
# 0.8145960252423514
# 0.8905291077014569
# 1.0
# 0.8115091428079356
# 0.8775634120991097
# 0.8395325736974772
# 0.9506983440908762
# 0.88184629454754
# 0.8115091428079356
# 1.0
# 0.759590996735251
# 0.8622109066850939
# 0.794650864741391
# 0.8692563822438262
# 0.8775634120991097
# 0.759590996735251
# 1.0
# -------------Recipes similarity---------------
# 1.0
# 0.9058970680531702
# 0.876188281410988
# 0.8921914246767313
# 0.9362319998716938
# 0.9077991339554589
# 0.9058970680531702
# 1.0
# 0.8960303314307113
# 0.8683352787585712
# 0.9251986105731227
# 0.9092774425049626
# 0.876188281410988
# 0.8960303314307113
# 1.0
# 0.8206932481018961
# 0.9234997668651656
# 0.9066167238062411
# 0.8921914246767313
# 0.8683352787585712
# 0.8206932481018961
# 1.0
# 0.8436152755971948
# 0.8890459855149918
# 0.9362319998716938
# 0.9251986105731227
# 0.9234997668651656
# 0.8436152755971948
# 1.0
# 0.8970581769256016
# 0.9077991339554589
# 0.9092774425049626
# 0.9066167238062411
# 0.8890459855149918
# 0.8970581769256016
# 1.0
# -------------Recipes similarity---------------
# 0.7908974953781049
# 0.6548518295341987
# 0.739868093247277
# 0.7337805432695084
# 0.6703983067394562
# 0.7674085842432804
# 0.7580808759364783
# 0.5323926147261138
# 0.711456026675044
# 0.7008472217256502
# 0.5443126469769464
# 0.7254376905581942
# 0.7884092498717231
# 0.5253234387230952
# 0.7214799557383781
# 0.6939840662542774
# 0.5243623425430298
# 0.7301757749509531
# 0.6633546838990971
# 0.4596805288222233
# 0.5643795549917598
# 0.6354896663817883
# 0.4868229640175464
# 0.6469780047472005
# 0.8458760451310995
# 0.6612351139999046
# 0.7954999302389468
# 0.7757727786024005
# 0.6793217450290933
# 0.7921867919801224
# 0.7706409888372543
# 0.5407034497147122
# 0.6932683264149205
# 0.7135981756961652
# 0.5491464776242874
# 0.7118486371023985

# Output with en_core_web_sm
# -------------Complaints similarity---------------
# 1.0
# D:\Dropbox\Dropbox\JM22110004707\Software Engineer Bootcamp\T38\example.py:38: UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
#   print(token.similarity(token_))
# 0.5090130466174222
# 0.7128344854285678
# 0.7586548283376252
# 0.5481749391003481
# 0.5452402796403908
# 0.5090130466174222
# 1.0
# 0.7270419360443711
# 0.44973754453402454
# 0.7358570174065904
# 0.539451287466797
# 0.7128344854285678
# 0.7270419360443711
# 1.0
# 0.7080282793889398
# 0.7212688611304713
# 0.6332525114389937
# 0.7586548283376252
# 0.44973754453402454
# 0.7080282793889398
# 1.0
# 0.5723389058792471
# 0.5394047113888343
# 0.5481749391003481
# 0.7358570174065904
# 0.7212688611304713
# 0.5723389058792471
# 1.0
# 0.49301711983715846
# 0.5452402796403908
# 0.539451287466797
# 0.6332525114389937
# 0.5394047113888343
# 0.49301711983715846
# 1.0
# -------------Recipes similarity---------------
# 1.0
# 0.7084920860284759
# 0.6028835452295952
# 0.7353469598150032
# D:\Dropbox\Dropbox\JM22110004707\Software Engineer Bootcamp\T38\example.py:58: UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
#   print(token.similarity(token_))
# 0.742543669049068
# 0.7203463346879786
# 0.7084920860284759
# 1.0
# 0.6627458558836178
# 0.6864292887388832
# 0.7183366047631227
# 0.7157533122977935
# 0.6028835452295952
# 0.6627458558836178
# 1.0
# 0.5845282964868923
# 0.6983784122312668
# 0.7327214060019888
# 0.7353469598150032
# 0.6864292887388832
# 0.5845282964868923
# 1.0
# 0.7533077379669134
# 0.7395232391726886
# 0.742543669049068
# 0.7183366047631227
# 0.6983784122312668
# 0.7533077379669134
# 1.0
# 0.8217725352900838
# 0.7203463346879786
# 0.7157533122977935
# 0.7327214060019888
# 0.7395232391726886
# 0.8217725352900838
# 1.0
# -------------Recipes similarity---------------
# 0.5966323790672934
# 0.3016252128392763
# D:\Dropbox\Dropbox\JM22110004707\Software Engineer Bootcamp\T38\example.py:69: UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
#   print(token.similarity(token_))
# 0.38263416407147594
# 0.5629846266692095
# 0.296555527068382
# 0.563792818513919
# 0.5024917576188229
# 0.1365427349914801
# 0.38694901428393064
# 0.5293672755251375
# 0.1609466229542559
# 0.5492883538552942
# 0.5717307684524712
# 0.23970944353302465
# 0.5015128759109356
# 0.5774550281213322
# 0.25644267374925117
# 0.47865004482282153
# 0.45971230146220154
# 0.13710956146501974
# 0.3076859397930708
# 0.5529147292867924
# 0.271413482850186
# 0.5467640587664471
# 0.6989333946320205
# 0.35674237652678187
# 0.5961972296632349
# 0.7574467156581833
# 0.5344295623356051
# 0.5674394925928256
# 0.6231520641071169
# 0.10303878804447938
# 0.4724182451659853
# 0.6803236382136475
# 0.30304246329307977
# 0.43135972184925

# en_core_web_sm has no vectors as indicated by the UserWarning message, subsequently leading to lower similarity
# scores than the ones obtained using en_core_web_md
