#!/bin/bash

###
### See https://www.icts.uiowa.edu/confluence/display/BRAINSPUBLIC/ANTS+conversion+to+antsRegistration+for+same+data+set
### For good parameters
###
#specify list of subjects
if [ ! -z $1 ]
then
	if [ -f $1 ]; then
		subjectList=`cat $1`
	else 
		subjectList="$1"
	fi
else
	subjectList=`cat subjects.txt`
fi

echo $subjectList
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8

# $false is false, 1 is true, cause bash
overwrite=$false

#specify folder names
experimentDir=`pwd`
inputDir=$experimentDir/preprocessed			#folder containing anatomical images
outDir=$experimentDir/processed			  #outputdir of normalized T1 files

#specify parameters for antsIntroduction.sh
#compulsory arguments
ImageDimension=3
OutPrefix='ANTS_'

#optional arguments
IgnoreHDRWarning=1
MaxIteration=40x90x60
N3Correct=0
QualityCheck=1
MetricType='PR'
TransformationType='GR'

ITS=" -i 100x100x30 " # 3 optimization levels
# different transformation models you can choose
TAFFINE=" -t Affine[0.1]"
TSYNWITHTIME=" -t SyN[0.25,5,0.01] " #" -r Gauss[3,0] " # spatiotemporal (full) diffeomorphism
TGREEDYSYN=" -t SyN[0.15,3.0,0.0] " #" -r Gauss[3,0] " # fast symmetric normalization 
TBSPLINESYN=" -t BSplineSyN[0.1,3.0,0.0] " #" -r Gauss[3,0] " # fast symmetric normalization 
TELAST=" -t Elast[1] -r Gauss[0.5,3] " # elastic
TEXP=" -t Exp[0.5,10] -r Gauss[0.5,3] " # exponential


#If not created yet, let's create a new output folder
if [ ! -d $outDir ]
then
	mkdir -p $outDir
fi

#go into the folder where the script should be run
cd $outDir

for subj in $subjectList
do
	echo '---------------------------------------------------------------------'
	echo $subj	
	T1="${subj}_T1_bet.nii.gz"
	MDWI="${subj}_MDWI_bet.nii.gz"	
	AP="${subj}_AP_bet.nii.gz"
	
	outRoot="$OutPrefix${subj}"
	outFinal="${outRoot}"

	#if anatomy of the subject wasn't normalized yet or if overwrite was set to 1=yes
	# the antsIntroduction script gets executed
	
	echo "$outDir/${outRoot}Warp.nii.gz"

	if [ ! -e "$outDir/${outRoot}1Warp.nii.gz" ] || [ $overwrite ]
	then

		FIXED="$inputDir/$T1"
		MOVING="$inputDir/$AP"		
		OUT="$outDir/$outRoot"

		# different metric choices for the user
		INTMSQ=" -m MSQ[${FIXED},${MOVING},1,0] "
		INTMI=" -m MI[${FIXED},${MOVING},1,32] "
		INTCC=" -m CC[${FIXED},${MOVING},1,3] "
		INTMATTS=" -m Mattes[${FIXED},${MOVING},1,32] "
		
		INT=${INTMI}
		TRANS=${TGREEDYSYN}
		# call antsIntroduction instead, deprecated
		#cmd="$ANTSPATH/antsIntroduction.sh -d $ImageDimension -r $inputDir/$T1 \
		#			 -i  -o $outDir/$outRoot \
		#			 -f $IgnoreHDRWarning -m $MaxIteration -n $N3Correct -q $QualityCheck \
		#			 -s $MetricType -t $TransformationType"
					 	
		# antsRegistration is preferred over ANTS, as it is faster
		#assemble the command for the script from the input parameters defined above
		# -m MI[fixed,moving,weight,num-of-histogram-bins] 
		#cmd="$ANTSPATH/ANTS $ImageDimension -o ${OUT} ${ITS} ${INT} ${TRANS}"
		

		
		cmd="time $ANTSPATH/antsRegistration -d $ImageDimension -o ${OUT} \
                     ${TAFFINE} ${INTMI} --convergence [10000x10000x10000x10000x10000] --shrink-factors 5x4x3x2x1 --smoothing-sigmas 4x3x2x1x0mm \
                     ${TGREEDYSYN} ${INTCC} --convergence [50x35x15,1e-7] --shrink-factors 3x2x1 --smoothing-sigmas 2x1x0mm  --use-histogram-matching 1 \
                      -z 1 "
                     
                     
        
  #     # cmd="time $ANTSPATH/antsRegistrationSyN.sh -d $ImageDimension -m ${MOVING} -f ${FIXED} -o ${OUT} -t s -n $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
		
		echo '#->'$cmd #state the command
		eval $cmd #execute the command

			
		 # here=`pwd`
		 # cd $outDir
		 # ConvertTransformFile $ImageDimension "${OUT}0GenericAffine.mat" "${OUT}Affine.txt"
		 # rename "s/${outRoot}1Warp/${outRoot}Warp/g" *.nii.gz		
		 # rename "s/${outRoot}1InverseWarp/${outRoot}InverseWarp/g" *.nii.gz				 
		 # cd $here
		
		
	else
		echo -e "Deformation info already exists"
	fi		
	
	#cmd="WarpImageMultiTransform 3 $inputDir/$MDWI $outDir/${outFinal}_FA.nii.gz -R $inputDir/$MDWI --use-BSpline $outDir/${outRoot}Warp.nii.gz $outDir/${outRoot}Affine.txt"
	cmd="antsApplyTransforms -d 3 -i $inputDir/$MDWI -r $inputDir/$MDWI -n BSpline -o $outDir/${outFinal}_deformed_DWS.nii.gz -t ${outRoot}1Warp.nii.gz -t ${outRoot}0GenericAffine.mat "
	echo '#->'$cmd
	eval $cmd
	
	#cmd="WarpImageMultiTransform 3 $inputDir/$MDWI $outDir/${outFinal}_T1.nii.gz -R $inputDir/$T1 --use-BSpline $outDir/${outRoot}Warp.nii.gz $outDir/${outRoot}Affine.txt"
	cmd="antsApplyTransforms -d 3 -i $inputDir/$MDWI -r $inputDir/$T1 -n BSpline -o $outDir/${outFinal}_deformed_T1.nii.gz -t ${outRoot}1Warp.nii.gz -t ${outRoot}0GenericAffine.mat "	
	echo '#->'$cmd
	eval $cmd

	cmd="antsApplyTransforms -d 3 -i $inputDir/$T1 -r $inputDir/$T1 -n BSpline -o $outDir/${outFinal}_invDeformed_T1.nii.gz 		-t [${outRoot}0GenericAffine.mat,1] -t ${outRoot}1InverseWarp.nii.gz"
	echo '#->'$cmd
	eval $cmd	

	cmd="antsApplyTransforms -d 3 -i $inputDir/$T1 -r $inputDir/$MDWI -n BSpline -o $outDir/${outFinal}_invDeformed_DWS.nii.gz 		-t [${outRoot}0GenericAffine.mat,1] -t ${outRoot}1InverseWarp.nii.gz"
	echo '#->'$cmd
	eval $cmd	
done
	


