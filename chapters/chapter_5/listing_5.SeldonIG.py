apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - name: step_one
          image: seldonio/step_one:1.0
        - name: step_two
          image: seldonio/step_two:1.0
        - name: step_three
          image: seldonio/step_three:1.0
    graph:
      name: step_one
      endpoint:
        type: REST
      type: MODEL
      children:
          name: step_two
          endpoint:
            type: REST
          type: MODEL
          children:
              name: step_three
              endpoint:
                type: REST
              type: MODEL
              children: []
    name: example
    replicas: 1
