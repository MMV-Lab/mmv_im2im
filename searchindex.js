Search.setIndex({docnames:["contributing","index","installation","mmv_im2im","mmv_im2im.bin","mmv_im2im.data_modules","mmv_im2im.models","mmv_im2im.postprocessing","mmv_im2im.preprocessing","mmv_im2im.utils","modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["contributing.rst","index.rst","installation.rst","mmv_im2im.rst","mmv_im2im.bin.rst","mmv_im2im.data_modules.rst","mmv_im2im.models.rst","mmv_im2im.postprocessing.rst","mmv_im2im.preprocessing.rst","mmv_im2im.utils.rst","modules.rst"],objects:{"":[[3,0,0,"-","mmv_im2im"]],"mmv_im2im.bin":[[4,0,0,"-","run_im2im"]],"mmv_im2im.bin.run_im2im":[[4,1,1,"","Args"],[4,2,1,"","main"]],"mmv_im2im.data_modules":[[5,0,0,"-","data_loader"],[5,0,0,"-","data_loader_embedseg"],[5,2,1,"","get_data_module"]],"mmv_im2im.data_modules.data_loader":[[5,1,1,"","Im2ImDataModule"]],"mmv_im2im.data_modules.data_loader.Im2ImDataModule":[[5,3,1,"","prepare_data"],[5,3,1,"","setup"],[5,3,1,"","test_dataloader"],[5,3,1,"","train_dataloader"],[5,3,1,"","val_dataloader"]],"mmv_im2im.data_modules.data_loader_embedseg":[[5,1,1,"","Im2ImDataModule"]],"mmv_im2im.data_modules.data_loader_embedseg.Im2ImDataModule":[[5,3,1,"","prepare_data"],[5,3,1,"","setup"],[5,3,1,"","test_dataloader"],[5,3,1,"","train_dataloader"],[5,3,1,"","val_dataloader"]],"mmv_im2im.models":[[6,0,0,"-","BranchedERFNet_2d"],[6,0,0,"-","BranchedERFNet_3d"],[6,0,0,"-","basic_FCN"],[6,0,0,"-","basic_GAN"],[6,0,0,"-","basic_embedseg"],[6,0,0,"-","basic_pix2pix"],[6,0,0,"-","erfnet"],[6,0,0,"-","erfnet_3d"],[6,0,0,"-","fnet_nn_3d_params"],[6,0,0,"-","layers_and_blocks"],[6,0,0,"-","pix2pixHD_generator_discriminator_2D"]],"mmv_im2im.models.BranchedERFNet_2d":[[6,1,1,"","BranchedERFNet_2d"]],"mmv_im2im.models.BranchedERFNet_2d.BranchedERFNet_2d":[[6,3,1,"","forward"],[6,3,1,"","init_output"],[6,4,1,"","training"]],"mmv_im2im.models.BranchedERFNet_3d":[[6,1,1,"","BranchedERFNet_3d"]],"mmv_im2im.models.BranchedERFNet_3d.BranchedERFNet_3d":[[6,3,1,"","forward"],[6,3,1,"","init_output"],[6,4,1,"","training"]],"mmv_im2im.models.basic_FCN":[[6,1,1,"","Model"]],"mmv_im2im.models.basic_FCN.Model":[[6,3,1,"","configure_optimizers"],[6,3,1,"","forward"],[6,3,1,"","prepare_batch"],[6,3,1,"","run_step"],[6,4,1,"","training"],[6,3,1,"","training_step"],[6,3,1,"","validation_step"]],"mmv_im2im.models.basic_embedseg":[[6,1,1,"","Model"]],"mmv_im2im.models.basic_embedseg.Model":[[6,3,1,"","configure_optimizers"],[6,3,1,"","forward"],[6,3,1,"","prepare_batch"],[6,3,1,"","run_step"],[6,4,1,"","training"],[6,3,1,"","training_step"],[6,3,1,"","validation_step"]],"mmv_im2im.models.basic_pix2pix":[[6,1,1,"","Model"]],"mmv_im2im.models.basic_pix2pix.Model":[[6,3,1,"","configure_optimizers"],[6,3,1,"","forward"],[6,3,1,"","run_step"],[6,4,1,"","training"],[6,3,1,"","training_epoch_end"],[6,3,1,"","training_step"],[6,3,1,"","validation_epoch_end"],[6,3,1,"","validation_step"]],"mmv_im2im.models.erfnet":[[6,1,1,"","Decoder"],[6,1,1,"","DownsamplerBlock"],[6,1,1,"","Encoder"],[6,1,1,"","Net"],[6,1,1,"","UpsamplerBlock"],[6,1,1,"","non_bottleneck_1d"]],"mmv_im2im.models.erfnet.Decoder":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet.DownsamplerBlock":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet.Encoder":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet.Net":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet.UpsamplerBlock":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet.non_bottleneck_1d":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet_3d":[[6,1,1,"","Decoder"],[6,1,1,"","DownsamplerBlock"],[6,1,1,"","Encoder"],[6,1,1,"","Net"],[6,1,1,"","UpsamplerBlock"],[6,1,1,"","non_bottleneck_1d"]],"mmv_im2im.models.erfnet_3d.Decoder":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet_3d.DownsamplerBlock":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet_3d.Encoder":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet_3d.Net":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet_3d.UpsamplerBlock":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.erfnet_3d.non_bottleneck_1d":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.fnet_nn_3d_params":[[6,1,1,"","Net"],[6,1,1,"","SubNet2Conv"]],"mmv_im2im.models.fnet_nn_3d_params.Net":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.fnet_nn_3d_params.SubNet2Conv":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.pix2pixHD_generator_discriminator_2D":[[6,1,1,"","Discriminator"],[6,1,1,"","Generator"],[6,1,1,"","PatchDiscriminator"],[6,1,1,"","ResidualBlock"]],"mmv_im2im.models.pix2pixHD_generator_discriminator_2D.Discriminator":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.pix2pixHD_generator_discriminator_2D.Generator":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.pix2pixHD_generator_discriminator_2D.PatchDiscriminator":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.models.pix2pixHD_generator_discriminator_2D.ResidualBlock":[[6,3,1,"","forward"],[6,4,1,"","training"]],"mmv_im2im.postprocessing":[[7,0,0,"-","basic_collection"],[7,0,0,"-","embedseg_cluster"]],"mmv_im2im.postprocessing.basic_collection":[[7,2,1,"","extract_segmentation"]],"mmv_im2im.postprocessing.embedseg_cluster":[[7,1,1,"","Cluster_2d"],[7,1,1,"","Cluster_3d"],[7,2,1,"","degrid"],[7,2,1,"","generate_instance_clusters"]],"mmv_im2im.postprocessing.embedseg_cluster.Cluster_2d":[[7,3,1,"","cluster"],[7,3,1,"","cluster_with_gt"]],"mmv_im2im.postprocessing.embedseg_cluster.Cluster_3d":[[7,3,1,"","cluster"],[7,3,1,"","cluster_with_gt"]],"mmv_im2im.preprocessing":[[8,0,0,"-","transforms"]],"mmv_im2im.preprocessing.transforms":[[8,2,1,"","dummy_to_ones"],[8,2,1,"","norm_around_center"],[8,2,1,"","to_float"]],"mmv_im2im.proj_tester":[[3,1,1,"","ProjectTester"]],"mmv_im2im.proj_tester.ProjectTester":[[3,3,1,"","run_inference"]],"mmv_im2im.proj_trainer":[[3,1,1,"","ProjectTrainer"]],"mmv_im2im.proj_trainer.ProjectTrainer":[[3,3,1,"","run_training"]],"mmv_im2im.utils":[[9,0,0,"-","embedding_loss"],[9,0,0,"-","embedseg_utils"],[9,0,0,"-","for_transform"],[9,0,0,"-","lovasz_losses"],[9,0,0,"-","misc"],[9,0,0,"-","piecewise_inference"],[9,0,0,"-","pix2pix_losses"]],"mmv_im2im.utils.embedding_loss":[[9,1,1,"","SpatialEmbLoss_2D"],[9,1,1,"","SpatialEmbLoss_3d"],[9,2,1,"","calculate_iou"]],"mmv_im2im.utils.embedding_loss.SpatialEmbLoss_2D":[[9,3,1,"","forward"],[9,4,1,"","training"]],"mmv_im2im.utils.embedding_loss.SpatialEmbLoss_3d":[[9,3,1,"","forward"],[9,4,1,"","training"]],"mmv_im2im.utils.embedseg_utils":[[9,2,1,"","fill_label_holes"],[9,2,1,"","generate_center_image"],[9,2,1,"","generate_center_image_2d"],[9,2,1,"","generate_center_image_3d"],[9,2,1,"","pairwise_python"]],"mmv_im2im.utils.for_transform":[[9,2,1,"","parse_tio_ops"]],"mmv_im2im.utils.lovasz_losses":[[9,1,1,"","StableBCELoss"],[9,2,1,"","binary_xloss"],[9,2,1,"","flatten_binary_scores"],[9,2,1,"","flatten_probas"],[9,2,1,"","iou"],[9,2,1,"","iou_binary"],[9,2,1,"","lovasz_grad"],[9,2,1,"","lovasz_hinge"],[9,2,1,"","lovasz_hinge_flat"],[9,2,1,"","lovasz_softmax"],[9,2,1,"","lovasz_softmax_flat"],[9,2,1,"","mean"],[9,2,1,"","xloss"]],"mmv_im2im.utils.lovasz_losses.StableBCELoss":[[9,3,1,"","forward"],[9,4,1,"","training"]],"mmv_im2im.utils.misc":[[9,2,1,"","aicsimageio_reader"],[9,2,1,"","generate_dataset_dict"],[9,2,1,"","generate_test_dataset_dict"],[9,2,1,"","get_max_shape"],[9,2,1,"","load_yaml_cfg"],[9,2,1,"","parse_config"],[9,2,1,"","parse_config_func"],[9,2,1,"","parse_config_func_without_params"],[9,2,1,"","parse_ops_list"]],"mmv_im2im.utils.piecewise_inference":[[9,2,1,"","predict_piecewise"]],"mmv_im2im.utils.pix2pix_losses":[[9,1,1,"","Pix2PixHD_loss"]],mmv_im2im:[[4,0,0,"-","bin"],[5,0,0,"-","data_modules"],[3,2,1,"","get_module_version"],[6,0,0,"-","models"],[7,0,0,"-","postprocessing"],[8,0,0,"-","preprocessing"],[3,0,0,"-","proj_tester"],[3,0,0,"-","proj_trainer"],[9,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","function","Python function"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:function","3":"py:method","4":"py:attribute"},terms:{"0":[5,6,7,9],"00028":6,"01":6,"02":6,"1":[5,6,7,9],"10":[6,7,9],"100":1,"1024":9,"128":7,"16":6,"1704":6,"1e":6,"2":[6,7,9],"2018":9,"28":5,"2d":[1,9],"3":[1,6,7,9],"32":[6,7,8,9],"36":7,"3d":[1,9],"4":6,"4d":[8,9],"5":[5,6,7],"6":6,"64":[6,9],"7":1,"768":7,"8":1,"9":6,"99":6,"abstract":1,"boolean":9,"break":1,"case":[1,5,6,9],"class":[3,4,5,6,7,9],"default":[5,6,7],"do":[5,6,7],"float":7,"function":[1,6,9],"int":[6,7,8,9],"new":[0,1],"public":[1,2],"return":[5,6,8,9],"short":0,"switch":6,"true":[5,6,9],"void":9,"while":[6,9],A:[0,1,5,6],At:6,BY:6,But:6,For:[1,5,9],If:[1,2,5,6],In:[5,6],It:[0,1,5,6],NOT:5,No:9,One:9,Or:2,The:[2,5,6],Then:[0,1],There:[1,5,6],To:2,With:6,__init__:5,_cm:9,_gt:9,_im:9,ab:6,about:5,abov:[5,6],acc:6,account:1,accuraci:6,act:6,action:1,actual:6,adam:6,add:[0,1,5,6],add_imag:6,addit:6,adjust:5,after:[1,6],afterward:[6,9],aicsimageio_read:9,alg:9,algorithm:6,all:[0,1,5,6,9],allencel:1,along:[6,9],also:[0,1,5,6],although:[6,9],alwai:[0,2],an:[1,5,6,9],anaconda:0,ani:[1,6,7],anisotropy_factor:9,anyth:6,append:6,appli:7,applic:1,appreci:0,approxim:9,ar:[0,1,5,6],arbitrari:5,arg:[4,5,6],argmax:6,argpars:4,argument:[5,6],around:8,arrai:[8,9],arxiv:6,assign:5,associ:6,atom:1,author:6,automat:6,avail:6,averag:[6,9],b:[0,9],back:6,backbon:1,background:9,backprop:6,backward:6,bad:5,bar:6,base:[1,3,4,5,6,7,9],basic_collect:[3,10],basic_embedseg:[3,10],basic_fcn:[3,10],basic_gan:[3,10],basic_pix2pix:[3,10],basic_xyz:1,batch:[5,6,9],batch_cifar:5,batch_idx:6,batch_mnist:5,batch_siz:5,becaus:1,been:[1,6],befor:[1,5],begin:5,being:6,below:6,berman:9,between:[6,9],bin:[3,10],binari:9,binary_xloss:9,bit:[0,6],bool:[6,9],both:[1,6,9],br:0,branch:0,branchederfnet_2d:[3,10],branchederfnet_3d:[3,10],browser:1,bucket:1,bugfix:0,build:[0,1,5],bump2vers:[0,1],bumpvers:1,c:[7,9],calcul:6,calculate_i:9,call:[1,5,6,9],callabl:9,callback:6,can:[0,1,2,5,6],care:[6,9],categori:1,cc:6,cd:0,cell:8,center:[8,9],center_imag:9,centroid:9,cfg:[3,5],chang:[0,1],chann:6,channel:[7,9],check:[0,1],checkout:0,choos:6,cifar:5,cifar_load:5,clean:1,clone:[0,2],closur:6,cluster:7,cluster_2d:7,cluster_3d:7,cluster_with_gt:7,cm:9,cmap:9,code:1,codecov:1,collect:5,column:9,com:[0,1,2,6],combin:9,command:2,commit:[0,1],compat:9,compos:5,comput:[6,9],condit:6,config:1,configur:[3,6],configure_optim:6,contain:[6,9],content:10,continu:6,contribut:1,control:6,cool:1,copi:2,core:[5,6],correct:5,correspond:[6,9],corrupt:5,cosineann:6,costmap_fn:9,could:6,cover:1,creat:[0,1],creativecommon:6,credit:0,cron:1,cross:9,csv:9,curl:2,current:[1,6],custom:6,cutoff:7,cycl:6,d:6,dai:1,data:[1,5,6,9],data_cfg:5,data_column:9,data_load:[1,3,10],data_loader_embedseg:[3,10],data_modul:[1,3,10],data_split:5,data_typ:9,dataload:[1,5,6],dataloader_i_output:6,dataloader_idx:[5,6],dataloader_out:6,dataloader_output_result:6,datamodul:5,dataset:[5,6],date:1,davi:6,davyneven:6,ddp:5,debug:1,decid:6,decod:6,deep:1,deepspe:6,def:[5,6],defin:[1,6,9],degrid:7,denois:1,depend:1,deploi:[1,4],depth:6,describ:6,descript:0,detail:0,determin:1,dev:[0,1],develop:0,devic:5,dict:[5,6,9],dictionari:[6,9],didn:6,differ:[1,6,9],dilat:6,dim:6,dimens:[6,8,9],dims_max:9,directli:1,directori:4,dis_opt:6,dis_sch:6,disabl:6,discrimin:6,displai:6,distribut:5,doc:1,document:5,don:[1,2,5,6],done:0,doubl:1,download:[2,5],download_data:5,downsamplerblock:6,dropdown:1,dropprob:6,dummy_to_on:8,dynam:5,dyx:9,e:[0,1,5,6],each:[1,5,6,9],easili:1,edit:[0,1],effect:1,effici:9,either:[2,5,7],els:[1,5],elsewher:9,embedding_loss:[3,10],embedseg:1,embedseg_clust:[3,10],embedseg_util:[3,10],empti:9,enabl:6,encod:[6,9],encourag:1,encrypt:1,end:6,enforc:6,enough:1,ensur:[1,5],entri:[1,3,6],entropi:9,environ:[0,1],epoch:6,equal:9,erfnet:[3,10],erfnet_3d:[3,10],error:9,esat:9,etc:[1,5],eval:6,even:1,everi:[0,1,5,6,9],everyth:1,ex:0,exampl:[1,5,6],example_config:1,example_imag:6,example_labelfre:1,exist:1,experi:6,exponentiallr:6,extens:9,extract:7,extract_segment:7,fals:[5,6,7,9],fancier:6,fast:9,fcn:1,featur:0,file:[0,1,5],filenam:9,fill:9,fill_label_hol:9,final_metr:6,final_valu:6,first:[1,6,9],fit:[5,6],fix:7,flatten:9,flatten_binary_scor:9,flatten_proba:9,fn:9,fnet_nn_3d_param:[3,10],folder:9,follow:[1,5],for_transform:[3,10],foreground:9,foreground_weight:9,fork:0,former:[6,9],forward:[6,9],found:[0,6],free:1,frequenc:6,from:[1,5,6,7],full:1,g:[1,5,6],gan:[1,6],gen_opt:6,gen_sch:6,gener:[1,6,9],generate_center_imag:9,generate_center_image_2d:9,generate_center_image_3d:9,generate_dataset_dict:9,generate_instance_clust:7,generate_synthetic_data:1,generate_test_dataset_dict:9,get:[4,6],get_data_modul:5,get_max_shap:9,get_module_vers:3,gh:[0,1],git:[0,2],github:[0,1,2,6],given:[0,6],global_rank:5,go:1,goe:6,good:5,gpu:6,gradient:[6,9],great:5,greatli:0,grid:6,grid_i:[7,9],grid_siz:7,grid_x:[7,9],grid_z:[7,9],ground:9,gt:9,gt_sort:9,guid:2,h:[7,9],ha:[1,6,9],handl:[0,1,6],happen:5,hardwar:5,have:[1,2,5,6],head:1,help:0,here:[0,1,5,6],hidden:6,hing:9,hole:9,hook:[5,6,9],hot:9,how:[0,6],howev:5,html:0,http:[1,2,6],i:1,id:9,ignor:[6,9],ignore_nan:9,im2im:[2,3],im2imdatamodul:5,im:[7,9],imag:[1,6,9],img:8,implement:[5,6],improv:6,in_channel:6,includ:[0,6,9],index:[1,6,8],individu:6,info:9,inform:1,infti:9,init_output:6,initi:[6,9],initialize_distribut:5,inner:6,input:[6,8,9],input_ch:6,input_channel:6,instal:[0,4],instanc:[6,7,9],instancenorm2d:6,instanti:1,instead:[5,6,9],integ:[5,6],intens:8,interest:6,intern:[6,9],interv:6,io:1,iou:9,iou_binari:9,ipu:6,item:6,its:6,jaccard:9,just:1,kei:[6,9],keyword:6,know:6,ku:9,kwarg:[6,9],l1:5,l:9,lab:[1,2],label:9,labelfre:1,labels_hat:6,last:6,latter:[6,9],launch:1,layers_and_block:[3,10],lbfg:6,lbl_img:9,lear:1,learn:6,learningratemonitor:6,len:6,length:[6,8],leuven:9,level:[1,3],licens:[1,6,9],lightin:1,lightn:[1,5,6],lightningdatamodul:5,lightningmodul:6,like:[1,6],line:1,linear:5,lint:[0,1],list:[1,5,6,9],litmodel:5,littl:0,load_data:5,load_yaml_cfg:9,loader:[1,5],loader_a:5,loader_b:5,loader_n:5,local:0,local_rank:5,locat:9,log:6,log_dict:6,logger:6,logic:5,logit:9,loss:[1,6,9],lot:1,lovasz:9,lovasz_grad:9,lovasz_hing:9,lovasz_hinge_flat:9,lovasz_loss:[3,10],lovasz_softmax:9,lovasz_softmax_flat:9,low:1,lr:6,lr_schedul:6,lr_scheduler_config:6,lstm:6,m2r:0,m:0,made:6,mai:1,main:[1,2,4,5],maintain:0,major:0,make:[0,1],make_grid:6,mani:6,mask:9,match:5,maxim:9,md:1,mean:[8,9],medoid:9,memori:1,mention:6,merg:1,meter:7,method:[2,5,6,7,9],metric:6,metric_to_track:6,metric_v:6,middl:1,might:6,min_mask_sum:7,min_object_s:7,min_unclustered_sum:7,min_z:8,minor:0,misc:[3,10],mit:[1,9],mmv:[2,3],mmv_im2im:[0,1,2],mnist:5,mnist_load:5,mode:[0,1,6,9],model:[1,3,5,10],model_di:6,model_gen:6,model_info_xx:[6,9],modul:[1,10],monitor:6,more:[1,5,9],most:[1,2,6],mult_chan:6,multi:[6,7,9],multipl:[5,6],must:6,n_batch:6,n_channel:6,n_critic:6,n_d:6,n_df:6,n_downsampl:6,n_gf:6,n_in:6,n_optim:6,n_out:6,n_residu:6,n_sigma:[6,7,9],name:[1,6],namespac:4,nanmean:9,natur:1,nc:6,ndarrai:7,necessari:5,need:[5,6,9],net:6,neven:6,next:[1,6],ninput:6,nn:[5,6,9],node:5,non:9,non_bottleneck_1d:6,none:[5,6,7,8,9],noqa:9,norm:6,norm_around_cent:8,norm_typ:6,normal:[5,6,8],note:[5,6],noutput:6,now:0,num_class:[5,6],number:6,numpi:[7,9],object:[3,6,7,9],often:6,ol:2,onc:[2,5],one:[1,5,6,9],one_hot:[7,9],onli:[5,6,9],only_encod:6,only_pres:9,open:1,oper:6,optim:6,optimizer_idx:6,optimizer_on:6,optimizer_step:6,optimizer_two:6,option:[6,7,8,9],order:[5,6],org:[1,6],origin:0,other:[0,1],otherwis:[1,9],out:6,out_channel:6,outer:6,output:6,output_ch:6,over:6,overlap:9,overrid:6,overridden:[6,9],own:[6,9],p:9,packag:[0,10],pad:6,padding_typ:6,page:1,pair:1,pairwise_python:9,paper:9,paramet:[1,3,6,8,9],paramref:[5,6],parent:4,parent_path:1,parse_config:9,parse_config_func:9,parse_config_func_without_param:9,parse_ops_list:9,parse_tio_op:9,pass:[0,6,9],password:1,patch:0,patchdiscrimin:6,path:[1,5,9],pathlib:9,pattern:5,per:[5,6,9],per_imag:9,perform:[6,9],piecewis:9,piecewise_infer:[3,10],pip:[0,1,2,4],pix2pix:1,pix2pix_loss:[3,10],pix2pixhd_generator_discriminator_2d:[3,10],pix2pixhd_loss:9,pixel:[8,9],pixel_i:[7,9],pixel_s:7,pixel_x:[7,9],pixel_z:[7,9],pleas:[1,5],plu:1,point:1,posit:5,possibl:[0,6],postprocess:[3,10],pr:1,precis:6,pred:[7,9],predict:[5,6,7,9],predict_kwarg:9,predict_piecewis:9,predictor:9,prefer:2,prepar:5,prepare_batch:6,prepare_data:5,prepare_data_per_nod:5,preprocess:[1,3,10],present:[6,9],previou:6,prior:1,proba:9,probabl:9,procedur:6,process:[2,5],produc:6,progress:6,proj_test:10,proj_train:[1,10],project:[0,1],projecttest:3,projecttrain:3,propag:6,pseudocod:6,psi:9,publish:0,pull:[0,1],push:[0,1],put:6,py:[1,2],pypi:[0,1],pypi_token:1,python:[0,1,2],pytorch:[1,5,6,9],pytorch_lightn:[5,6],quickli:6,quilt:1,r:9,ram:9,rare:1,rate:6,raw:0,re:0,read:0,readi:0,realli:1,recent:2,recip:[6,9],recommend:[0,1,5,6],reducelronplateau:6,reflect:6,regist:[1,6,9],relat:1,releas:[0,1],reload:5,reload_dataloaders_every_n_epoch:5,remind:0,remov:9,repo:[0,1,2],repositori:[1,2],repres:9,request:[0,1,5],requir:[1,6],residualblock:6,resolv:0,respect:8,result:[5,9],retain:1,root:5,run:[0,1,2,6,9],run_im2im:[1,3,10],run_infer:3,run_step:6,run_train:3,s:[0,5,6,8],safe:5,same:[1,6],sampl:[4,5],sample_img:6,sampler:5,save:5,schedul:6,score:9,script:[1,4],scriptmodul:[6,9],search:1,second:6,secret:1,section:[1,5],see:[1,5,6,9],seed_thresh:7,segment:7,select:[1,7],self:[5,6],separ:1,sequenc:5,sequenti:6,set:[0,1,5,6],setup:[2,5],sgd:6,shape:9,share:[1,5,6,9],should:[1,6,9],shown:6,shuffl:5,side:1,sign:1,silent:[6,9],similar:6,simpli:6,sinc:[5,6,9],singl:[5,6],size:[6,9],skimag:7,skip:6,slice:9,small:[1,9],smooth:6,so:[1,5,6],softmax:9,some:[1,6],some_other_st:5,some_st:5,someth:[5,6],sort:9,sourc:[1,3,4,5,6,7,8,9],source_fn:9,spatialembed:6,spatialembloss_2d:9,spatialembloss_3d:9,specif:[1,6],specifi:[1,5,6,9],speed_up:9,split:[5,9],squash:1,squeez:6,stabl:1,stablebceloss:9,stage:5,state:[5,6,9],std:8,step:6,still:1,stop:6,store:1,str:[7,9],strict:6,string:9,strongli:1,strss:9,style:9,sub:[8,9],subclass:[6,9],subject:9,submit:0,submodul:10,subnet2conv:6,subpackag:10,sum:6,support:6,sure:[0,1],system:5,t:[1,2,5,6,9],t_max:6,tab:1,tag:[0,1],take:[6,9],tarbal:2,target:9,target_fn:9,tbptt_step:6,tell:6,tensor:[5,6,7,8,9],tensor_in:9,termin:2,test:[0,1,5],test_dataload:5,test_step:5,text:6,thei:0,them:[5,6,9],thi:[0,1,2,4,5,6,9],thing:6,those:6,three:[1,9],through:[0,2,6],thu:6,tiff:[0,9],time:[1,6],to_float:8,token:5,top:3,torch:[5,6,7,8,9],torchvis:6,total:5,totensor:5,tox:1,tpu:6,train:[1,3,5,6,9],train_batch:6,train_data:6,train_dataload:5,train_out:6,trainer:[1,5],training_epoch_end:6,training_step:6,training_step_output:6,trans_func:9,transform:[2,3,5,10],truncat:6,truncated_bptt_step:6,truth:9,tupl:6,turn:1,two:[5,6,9],type:[8,9],under:[1,6],union:[7,9],uniqu:9,unit:6,univers:1,unless:5,unpair:1,up:[0,1],updat:6,upsamplerblock:6,us:[1,4,5,6,9],user:4,util:[1,3,5,6,10],val:6,val_acc:6,val_batch:6,val_data:6,val_dataload:5,val_loss:6,val_out:6,val_step_output:6,valid:[5,6],validation_epoch_end:6,validation_stag:6,validation_step:[5,6],validation_step_end:6,valu:[1,6,7],variabl:9,variou:1,veri:1,version:[0,8],via:1,view:1,virtualenv:[0,4],visit:1,w605:9,w:[7,9],w_inst:9,w_seed:9,w_var:9,wai:5,warn:6,wasserstein:6,we:[1,6],web:1,websit:0,welcom:0,well:1,what:6,whatev:6,when:[0,1,4,5,6],where:[5,9],wherea:6,which:[1,5,6,7,9],whose:6,within:[5,6,9],without:6,won:6,work:[0,1],worri:1,wrapper:1,x:[5,6,7,9],xloss:9,y:[5,6,7,9],yaml:1,yaml_path:9,you:[0,2,5,6],your:[0,1,2,5,6],your_development_typ:0,your_name_her:0,yourself:[1,5],yx:9,z:[6,7,8,9],z_center:8,zero:9},titles:["Contributing","Welcome to MMV Im2Im Transformation\u2019s documentation!","Installation","mmv_im2im package","mmv_im2im.bin package","mmv_im2im.data_modules package","mmv_im2im.models package","mmv_im2im.postprocessing package","mmv_im2im.preprocessing package","mmv_im2im.utils package","mmv_im2im"],titleterms:{The:1,To:1,addit:1,basic_collect:7,basic_embedseg:6,basic_fcn:6,basic_gan:6,basic_pix2pix:6,bin:4,branch:1,branchederfnet_2d:6,branchederfnet_3d:6,command:1,content:[3,4,5,6,7,8,9],contribut:0,data_load:5,data_loader_embedseg:5,data_modul:5,deploi:0,design:1,develop:1,document:1,embedding_loss:9,embedseg_clust:7,embedseg_util:9,erfnet:6,erfnet_3d:6,featur:1,fnet_nn_3d_param:6,for_transform:9,four:1,from:2,get:0,git:1,im2im:1,indic:1,instal:[1,2],know:1,layers_and_block:6,lovasz_loss:9,misc:9,mmv:1,mmv_im2im:[3,4,5,6,7,8,9,10],model:6,modul:[3,4,5,6,7,8,9],need:1,note:1,option:1,packag:[1,3,4,5,6,7,8,9],piecewise_infer:9,pix2pix_loss:9,pix2pixhd_generator_discriminator_2d:6,postprocess:7,preprocess:8,proj_test:3,proj_train:3,quick:1,releas:2,run_im2im:4,s:1,setup:1,sourc:2,stabl:2,start:[0,1],step:1,strategi:1,submodul:[3,4,5,6,7,8,9],subpackag:3,suggest:1,tabl:1,transform:[1,8],util:9,welcom:1,you:1}})