import configparser

env_config = configparser.ConfigParser()

env_config.read('Ryan-Reach-SO-ARM101-Normalized-v0.ini')

out = env_config['policy']['hidden_dims']

print(out)

test = [int(x) for x in out.split(',')]

print(test)