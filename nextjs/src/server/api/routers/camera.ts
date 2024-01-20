import { z } from "zod";

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";

export const cameraRouter = createTRPCRouter({
  setStatus: protectedProcedure
    .input(z.object({ cameraId: z.number(), status: z.string() }))
    .mutation(async ({ ctx, input }) => {
      console.log("input", input);

      let camera = await ctx.db.camera.findFirst({
        where: {
          cameraId: input.cameraId,
          user: { id: ctx.session.user.id },
        },
      });

      if (!camera) {
        camera = await ctx.db.camera.create({
          data: {
            cameraId: input.cameraId,
            user: { connect: { id: ctx.session.user.id } },
          },
        });
      }
      console.log("cameraId: ", camera.cameraId);

      let status = await ctx.db.status.findFirst({
        where: {
          slug: input.status,
        },
      });

      if (!status) {
        status = await ctx.db.status.create({
          data: {
            name: input.status,
            slug: input.status.toLowerCase().replace(/\s+/g, "-"),
          },
        });
      }

      if (status) {
        console.log("status: ", status);
      }

      camera = await ctx.db.camera.update({
        where: {
          id: camera.id,
        },
        data: {
          status: { connect: { id: status.id } },
        },
        include: {
          status: true,
        },
      });

      return camera;
    }),
});

//   hello: publicProcedure
//     .input(z.object({ text: z.string() }))
//     .query(({ input }) => {
//       return {
//         greeting: `Hello ${input.text}`,
//       };
//     }),

//   create: protectedProcedure
//     .input(z.object({ name: z.string().min(1) }))
//     .mutation(async ({ ctx, input }) => {
//       // simulate a slow db call
//       await new Promise((resolve) => setTimeout(resolve, 1000));

//       return ctx.db.post.create({
//         data: {
//           name: input.name,
//           createdBy: { connect: { id: ctx.session.user.id } },
//         },
//       });
//     }),

//   getLatest: protectedProcedure.query(({ ctx }) => {
//     return ctx.db.post.findFirst({
//       orderBy: { createdAt: "desc" },
//       where: { createdBy: { id: ctx.session.user.id } },
//     });
//   }),

//   getSecretMessage: protectedProcedure.query(() => {
//     return "you can now see this secret message!";
//   }),
// });
