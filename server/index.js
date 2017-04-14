const {join} = require('path')
const serve = require('serve')

const path = join(__dirname,'..','client')
serve(path, {
  port: 3000,
  ignore: ['node_modules']
})
