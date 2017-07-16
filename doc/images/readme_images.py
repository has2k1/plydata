import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_line
from plydata import define, define_where

kwargs = dict(width=6, height=4)

df = pd.DataFrame({'x': np.linspace(0, 2*np.pi, 500)})

p = (df
     >> define(y='np.sin(x)')
     >> define_where('y>=0', sign=('"positive"', '"negative"'))
     >> (ggplot(aes('x', 'y'))
         + geom_line(aes(color='sign'), size=1.5))
     )
p.save('readme-image.png', **kwargs)
