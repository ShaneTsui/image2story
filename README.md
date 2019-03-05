# image2story

This project is a pytorch reimplementation of [neural-storyteller
](https://github.com/ryankiros/neural-storyteller).

# Design
The project consists of 4 sub-modules, which are
- CNN feature extractor
- image-sentence embedding
- skip-thought vector encoder
- RNN decoder

Specially, we used style shifting trick between embedding and encoder, which allows our model to transfer standard image captions to the style of stories from novels.