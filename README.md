# Kaggle competiton. Motion Prediction: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles

Just realised this is all about computational muscle which I don't have (laptop with an old GTX980) but I think I can use my knowledge of vehicle dynamics.

Doing research it seems vectornet from waymo (which probably are ahead of the rest big dogs in autonomous driving) is the best approach although it's difficult to say
they did their homework to benchmark the model against others:

https://arxiv.org/pdf/2005.04259.pdf

I don't feel confortable using graph nets as I've never coded them myself. So I'll leave that for the future.

There are two approaches, using the reasterised image or taking a representation of the roads with lines.
Uber and Lyft would say they tend to use rasterisers. A good example is this paper:
https://arxiv.org/pdf/1809.10732.pdf

The competition baseline is just a CNN and dense NN in series, similar approach to the paper above. 

Looking at the winner of similar competitions (argoverse and nuscenes, https://github.com/nutonomy/nuscenes-devkit), rasterisers might not be the way to go. 
They all seem to have in common a LSTM encoding approach. Mostly based on this "old" paper:

http://vision.stanford.edu/pdf/alahi2016cvpr.pdf

Some examples of winner from the other competitions:

https://arxiv.org/pdf/1910.03650.pdf

http://cvrr.ucsd.edu/publications/2020/MHAJAM.pdf

https://arxiv.org/pdf/1904.04776.pdf


You can check the video below from minute 15 or so:

https://www.youtube.com/watch?v=Vcbj_peZT4Q&feature=emb_title

You probably noticed they all use attention mechanisms.

I'll try to put something together using some of the things above. I believe they need to augmentate the car state vector to add more detail.


