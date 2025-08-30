import { QuartzTransformerPlugin } from "../types"

function remarkSuperSub() {
  return (tree: any) => {
    function transform(node: any) {
      if (node.type === 'text') {
        // Transform ^text^ to superscript
        node.value = node.value.replace(/\^([^^]+)\^/g, '<sup>$1</sup>')
        // Transform ~text~ to subscript  
        node.value = node.value.replace(/~([^~]+)~/g, '<sub>$1</sub>')
      }
      
      if (node.children) {
        node.children.forEach(transform)
      }
    }
    
    transform(tree)
  }
}

export const SuperscriptSubscript: QuartzTransformerPlugin = () => {
  return {
    name: "SuperscriptSubscript",
    markdownPlugins() {
      return [remarkSuperSub]
    },
  }
}
