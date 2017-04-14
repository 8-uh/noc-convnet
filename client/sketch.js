const progress = () => model.getLoadingProgress()
const log = (...args) => console.log(...args)
let model = null


var isLoading = true



function setup() {
  background(25)

  fill(255)
  createCanvas(window.innerWidth, window.innerHeight)
  model = new KerasJS.Model({
    gpu: true,
    filepaths: {
      weights: '/data/model_weights.buf',
      metadata: '/data/model_metadata.json',
      model: '/data/model.json'
    }
  })
}

function draw() {
  push()
  if(isLoading && progress() < 100) {
    //spinner()
    isLoading = progress() < 100
    log('load progress:', progress())

  } else {
    console.log ('wat?')
  }
  pop()
}

function spinner() {
  push()
  strokeWeight(4)
  stroke(255)
  rotate((frameCount * 0.01) * TWO_PI)
  arc(0, 0, height * 0.25, height * 0.25, 0, TWO_PI * (progres() * 0.01))
  pop
}
