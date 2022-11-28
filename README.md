# social-value-orientation
RL models of social value orientation in the trust game using DQN, IBL, and NEF agents

 - add to pipfile:

	[[source]]
	url = "https://pypi.org/simple"
	verify_ssl = true
	name = "pypi"

	[[source]]
	url = "https://download.pytorch.org/whl/cpu/"
	verify_ssl = false
	name = "pytorch"

	[dev-packages]

	[packages]
	torch = {version="*", index="pytorch"}
	numpy = {version="*"}

 - pipenv install --index=pytorch torch
