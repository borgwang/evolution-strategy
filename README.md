# evolution-strategy

This repo implemented some popular evolution strategies including:

- [Simple Evolution Strategy](https://en.wikipedia.org/wiki/Evolution_strategy)
- [Genetic Algorithm (GA)](http://www.boente.eti.br/fuzzy/ebook-fuzzy-mitchell.pdf)
- [Covariance-Matrix Adaptation Evolution Strategy (CMA)](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)
- [Parameter-Exploring Policy Gradient (PEPG)](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=A64D1AE8313A364B814998E9E245B40A?doi=10.1.1.180.7104&rep=rep1&type=pdf)
- [OpenAI ES (A variant of PEPG)](https://arxiv.org/abs/1703.03864)

## Requirements
- Linux or Mac OS
- python 3.x

## Install

> pip install -r requirements.txt

## Run
For training:

> python train.py bullet_racecar -n 8 -t 4 -e pepg

For model evaluating:

> python eval_model.py bullet_racecar -m /path/to/model


## Related resources
- This repo is built with some modifications to [estool](https://github.com/hardmaru/estool).
- These two [awesome](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) [posts](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/).
