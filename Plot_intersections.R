# visualize set intersections
library("reticulate")
library("SuperExactTest")

threshold_0 = 768*1 
threshold_1 = 768*2
threshold_2 = 768*3
threshold_3 = 768*4
threshold_4 = 768*5
threshold_5 = 768*6
threshold_6 = 768*7
threshold_7 = 768*8 
threshold_8 = 768*9
threshold_9 = 768*10
threshold_10 = 768*11
threshold_11 = 768*12

##################
# SHORT-DISTANCE #
##################

s_100 <- list()

german <- unlist(py_load_object("de_simple_agreement/ordering"))
german <- paste(german[threshold_7 < german & german < threshold_8][1:100])
s_100$German <- german
min(s_100$German)

english <- unlist(py_load_object("en_simple_agreement/ordering"))
english <- paste(english[threshold_7 < english & english < threshold_8][1:100])
s_100$English <- english

hebrew <- unlist(py_load_object("he_simple_agreement/ordering"))
hebrew <- paste(hebrew[threshold_7 < hebrew & hebrew < threshold_8][1:100])
s_100$Hebrew <- hebrew

french <- unlist(py_load_object("fr_simple_agreement/ordering"))
french <- paste(french[threshold_7 < french & french < threshold_8][1:100])
s_100$French <- french

russian <- unlist(py_load_object("ru_simple_agreement/ordering"))
russian <- paste(russian[threshold_7 < russian & russian < threshold_8][1:100])
s_100$Russian <- russian

total=768
res=supertest(s_100, n=total)
plot(res, sort.by="size", margin=c(2,2,2,2), color.scale.pos=c(0.85,1), legend.pos=c(0.86,0.15))
summary(res)

#################
# LONG-DISTANCE #
#################

vp_100 <- list()

german <- unlist(py_load_object("de_long_vp_coord/ordering"))
german <- paste(german[threshold_8 < german & german < threshold_9][1:100])
vp_100$German <- german
min(vp_100$German)

english <- unlist(py_load_object("en_long_vp_coord/ordering"))
english <- paste(english[threshold_8 < english & english < threshold_9][1:100])
vp_100$English <- english

hebrew <- unlist(py_load_object("he_long_vp_coord/ordering"))
hebrew <- paste(hebrew[threshold_8 < hebrew & hebrew < threshold_9][1:100])
vp_100$Hebrew <- hebrew

french <- unlist(py_load_object("fr_long_vp_coord/ordering"))
french <- paste(french[threshold_8 < french & french < threshold_9][1:100])
vp_100$French <- french

russian <- unlist(py_load_object("ru_long_vp_coord/ordering"))
russian <- paste(russian[threshold_8 < russian & russian < threshold_9][1:100])
vp_100$Russian <- russian

total=768
res=supertest(vp_100, n=total)
plot(res, sort.by="size", margin=c(2,2,2,2), color.scale.pos=c(0.85,1), legend.pos=c(0.86,0.15))
summary(res)

