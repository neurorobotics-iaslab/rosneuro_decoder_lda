# ROS-Neuro lda decoder package

This package implements a LDA classifier as a plugin for rosneuro::decoder::Decoder and as class. The test used it as class, for the usage as rosnode it misses the tensor message after the pwelch computation.

## Usage
The package required as ros parameter:
<ul>
    <li> <b>cfg_name</b>: which is the name of the structure in the yaml file </li>
    <li> <b>yaml file</b>: which contains the structure for the lda classifier </li>
</ul>

## Example of yaml file
```
LdaCfg:
  name: "lda"
  params:
    filename: "file1"
    subject: "s1"
    n_classes: 2
    class_lbs: [771, 773]
    n_features: 5
    idchans: [1, 2]
    freqs: "10 12;
            20 22 24;"
    priors: [0.5, 0.5]
    lambda: 0.5
    means: "0.4964 1.4994;
            0.5297 1.5036;
            0.4903 1.5054;
            0.4491 1.4950;
            0.4956 1.4733;" 
    covs: "1.1340 0.1117 0.1027 0.1137 0.1006;
           0.1117 1.1363 0.1144 0.1189 0.1126;
           0.1027 0.1144 1.1470 0.1132 0.1099;
           0.1137 0.1189 0.1132 1.1372 0.1028;
           0.1006 0.1126 0.1099 0.1028 1.1131;"
```

Some parameters are hard coded:
<ul>
    <li> <b>idchans</b>: the index of the channels from 1 to the number of channels used; </li>
    <li> <b>freqs</b>: the selected frequencies; </li>
    <li> <b>means</b>: matrix of [features x classes]; </li>
    <li> <b>covs</b>: matrix of [features x features]. </li>
</ul>