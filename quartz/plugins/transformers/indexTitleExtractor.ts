import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"
import { toString } from "mdast-util-to-string"

export const IndexTitleExtractor: QuartzTransformerPlugin = () => {
  return {
    name: "IndexTitleExtractor",
    markdownPlugins() {
      return [
        () => {
          return async (tree: Root, file) => {
            // Only process files named "index"
            if (file.stem === "index") {
              // Extract the first H1 heading
              let h1Text: string | null = null
              visit(tree, "heading", (node) => {
                if (node.depth === 1 && !h1Text) {
                  h1Text = toString(node)
                }
              })
              
              // If we found an H1 and there's no explicit frontmatter title, use the H1
              if (h1Text && (!file.data.frontmatter?.title || file.data.frontmatter.title === "index")) {
                if (file.data.frontmatter) {
                  file.data.frontmatter.title = h1Text
                } else {
                  file.data.frontmatter = { title: h1Text }
                }
              }
            }
          }
        },
      ]
    },
  }
}
