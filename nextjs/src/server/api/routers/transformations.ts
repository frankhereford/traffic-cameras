import { z } from "zod"
import * as fs from "fs"
import * as os from "os"
import * as path from "path"
import { v4 as uuidv4 } from "uuid"
import { spawn } from "child_process"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

type BoundingBox = {
  Width: number
  Height: number
  Left: number
  Top: number
}

type Instance = {
  BoundingBox: BoundingBox
  Confidence: number
}

type Category = {
  Name: string
}

type Parent = {
  Name: string
}

type Alias = {
  Name: string
}

type Label = {
  Name: string
  Confidence: number
  Instances: Instance[]
  Parents: Parent[]
  Aliases: Alias[]
  Categories: Category[]
}

type ResponseMetadata = {
  RequestId: string
  HTTPStatusCode: number
  HTTPHeaders: {
    "x-amzn-requestid": string
    "content-type": string
    "content-length": string
    date: string
  }
  RetryAttempts: number
}

type ImageRecognitionResponse = {
  Labels: Label[]
  LabelModelVersion: string
  ResponseMetadata: ResponseMetadata
}

export const transformation = createTRPCRouter({
  submitWarpRequest: publicProcedure
    .input(
      z.object({
        image: z.string(),
        points: z.array(
          z.object({
            cctvPoint: z.object({
              x: z.number(),
              y: z.number(),
            }),
            mapPoint: z.object({
              lat: z.number(),
              lng: z.number(),
            }),
          }),
        ),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      // console.log("input", JSON.stringify(input, null, 2))

      const baseDir = path.join(os.tmpdir(), "transformations")
      if (!fs.existsSync(baseDir)) {
        fs.mkdirSync(baseDir)
      }

      const uuid = uuidv4()
      const tmpDir = path.join(baseDir, uuid)
      fs.mkdirSync(tmpDir, { recursive: true })

      console.log("Temporary directory created.", tmpDir)

      // Remove the base64 image prefix
      const base64Data = input.image.replace(/^data:image\/jpeg;base64,/, "")

      // Convert the base64 string to a Buffer
      const dataBuffer = Buffer.from(base64Data, "base64")

      // Write the Buffer to a file
      const imagePath = path.join(tmpDir, "cctvImage.jpg")
      fs.writeFileSync(imagePath, dataBuffer)
      // console.log("Image written to disk at", imagePath)
      // Convert the points data to a JSON string
      const pointsData = JSON.stringify(input.points, null, 2)

      // Write the JSON string to a file
      const pointsPath = path.join(tmpDir, "points.json")
      fs.writeFileSync(pointsPath, pointsData)

      const labels = new Promise((resolve, reject) => {
        const pythonProcess = spawn("python3", [
          "/transformer/image_labeler.py",
          uuid,
        ])

        let output = ""
        pythonProcess.stdout.on("data", (data) => {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
          output += data.toString()
        })

        pythonProcess.on("close", (code) => {
          if (code !== 0) {
            return reject(new Error(`Python process exited with code ${code}`))
          }

          // console.log("output", output)

          let recognition_result
          try {
            recognition_result = JSON.parse(output) as ImageRecognitionResponse
          } catch (err) {
            return reject(new Error("Failed to parse Python output as JSON"))
          }

          resolve(recognition_result)
        })
      })

      console.log("recognition_result", await labels)

      console.log("done")

      return labels
    }),

  getSecretMessage: protectedProcedure.query(() => {
    return "you can now see this secret message!"
  }),
})
