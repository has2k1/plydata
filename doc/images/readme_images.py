import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_line
from plydata import define

kwargs = dict(width=6, height=4)

df = pd.DataFrame({'x': np.linspace(0, 2*np.pi, 100)})

p = df >> define(y='np.sin(x)') >> ggplot(aes('x', 'y')) + geom_line()
p.save('readme-image.png', **kwargs)
