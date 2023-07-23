# from typing import List
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# def plot_token_rewards(tokens: List[str], rewards: List[float]):

#     assert len(rewards) == len(tokens), "Rewards and tokens must have the same length"

#     # Create a colormap
#     cmap = mcolors.LinearSegmentedColormap.from_list("n",["red", "blue", "green"])
#     norm = mcolors.Normalize(vmin=min(rewards), vmax=max(rewards))
#     colors = cmap(norm(rewards))

#     plt.figure(figsize=(15, 5))
#     plt.title('Rewards per Token')
#     bars = plt.bar(tokens, rewards, color=colors)
#     plt.xlabel('Tokens')
#     plt.ylabel('Reward Value')
#     plt.xticks(rotation=45)

#     # Adding a colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     plt.colorbar(sm)
#     plt.show()

from typing import List, Union
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import numpy as np

def plot_token_rewards(tokens: List[str], raw_rewards: Union[float, List[float]], title='Rewards per Token'):

    rewards: List[float]
    if isinstance(raw_rewards, (int, float)):
        rewards_tmp: List[float] = [0.0] * len(tokens)
        rewards_tmp[-1] = raw_rewards
        rewards = rewards_tmp
    else:
        rewards = raw_rewards

    assert len(rewards) == len(tokens), f"Rewards and tokens must have the same length. len(rewards)={len(rewards)}, len(tokens)={len(tokens)}"

    total_rewards: float = round(np.sum(rewards), 2)

    # Create a colormap
    # cmap = mcolors.LinearSegmentedColormap.from_list("n",["red", "blue", "green"])
    # norm = mcolors.Normalize(vmin=min(rewards), vmax=max(rewards))
    # norm = mcolors.Normalize(vmin=-1, vmax=1)
    # colors = cmap(norm(rewards))

    # Convert matplotlib colors to plotly colors (in the 'rgb(r, g, b)' format)
    # plotly_colors = ['rgb'+str(tuple(int(255 * c) for c in color[:3])) for color in colors]

    # clean_tokens = [f"'{token.replace('Ġ', ' ')}'" for token in tokens]
    # prepend index to clean_token
    # clean_tokens = [f"{i}: {token}" for i, token in enumerate(clean_tokens)]
    clean_tokens = [f"{i}: {token}" for i, token in enumerate(tokens)]

    fig = go.Figure(data=[go.Bar(
        # x=tokens, 
        x=clean_tokens, 
        y=rewards, 
        # marker_color=plotly_colors, # use converted plotly colors
        marker=dict(
            color=rewards, # set color to an array/list of desired values
            # colorscale='RdBu', # choose a colorscale
            # red-green colorscale
            colorscale=[
                [0, "rgb(255, 0, 0)"],
                # [0.5, "rgb(0, 0, 255)"],
                [1, "rgb(0, 255, 0)"]
            ],
            cmin=-1, # set the lower bound of color domain
            cmax=1, # set the upper bound of color domain
        ),
        hovertemplate = 'Token: %{x}<br>Reward: %{y}<extra></extra>' # custom hovertemplate
    )])

    fig.update_layout(
        # title=title,
        title=f"{title}<br>Total: {total_rewards}",
        xaxis_title='Tokens',
        # yaxis_title=f"Reward Value<br>Total: {total_rewards}",
        yaxis_title="Reward Value",
        yaxis = dict(range=[-1,1]), # forcing y-axis to be in range -1 to +1
        autosize=False,
        width=1000,
        height=500,
    )

    fig.show()
