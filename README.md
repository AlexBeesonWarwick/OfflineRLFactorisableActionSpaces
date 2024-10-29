# An investigation of offline reinforcement learning in factorisable action spaces
UNDER CONSTRUCTION

Expanding reinforcement learning (RL) to offline domains generates promising prospects, particularly in sectors where data collection poses substantial challenges or risks.  Pivotal to the success of transferring RL offline is mitigating overestimation bias in value estimates for state-action pairs absent from data. Whilst numerous approaches have been proposed in recent years, these tend to focus primarily on continuous or small-scale discrete action spaces.  Factorised discrete action spaces, on the other hand, have received relatively little attention, despite many real-world problems naturally having factorisable actions.  In this work, we undertake a formative investigation into offline reinforcement learning in factorisable action spaces.  Using value-decomposition as formulated in DecQN as a foundation, we present the case for a factorised approach and conduct an extensive empirical evaluation of several offline techniques adapted to the factorised setting.  In the absence of established benchmarks, we introduce a suite of our own comprising datasets of varying quality and task complexity.  Advocating for reproducible research and innovation, we make all datasets available for public use, alongside our code base.
