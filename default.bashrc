############### MY CONFIGS ###############
alias grep='grep --color=auto -i'
alias subjd='setenv SUBJECTS_DIR `pwd`'
alias ls='ls --group-directories-first --color=auto'
alias pwdc='pwd | tr "\n" " " | xsel -bi'

source $FREESURFER_HOME/SetUpFreeSurfer.sh
source /etc/fsl/5.0/fsl.sh