from setuptools import setup

with open('requirements.txt', 'r', encoding='utf-8') as r:
    requirements = [line.strip() for line in r]

setup(name='genplan-strategy-refine',
      packages=['agents', 'agentic_genplan', 'feedback_generators', 'llm_models', 'utils', 'evaluation'],
      install_requires=requirements
      )
